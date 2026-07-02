use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray, ListArray,
    ListBuilder, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;

use crate::core::{AllocationOutput, Demand, DisruptionOutput, Edge, EdgeFlow, Loss, OdFlow};

pub fn read_network_ipc(bytes: &[u8]) -> Result<Vec<Edge>, String> {
    read_network_batches(&read_batches(bytes)?)
}

pub fn read_demands_ipc(bytes: &[u8]) -> Result<Vec<Demand>, String> {
    read_demands_batches(&read_batches(bytes)?)
}

pub fn read_od_flows_ipc(bytes: &[u8]) -> Result<Vec<OdFlow>, String> {
    read_od_flows_batches(&read_batches(bytes)?)
}

pub fn read_network_batches(batches: &[RecordBatch]) -> Result<Vec<Edge>, String> {
    let mut edges = Vec::new();
    for batch in batches {
        let from = required_column(batch, "edge_from")?;
        let to = required_column(batch, "edge_to")?;
        let id = required_column(batch, "edge_id")?;
        for row in 0..batch.num_rows() {
            edges.push(Edge {
                from: integer_value(from.as_ref(), "edge_from", row)?,
                to: integer_value(to.as_ref(), "edge_to", row)?,
                id: integer_value(id.as_ref(), "edge_id", row)?,
                cost: numeric_value_or_default(batch, "cost", row, 1.0)?,
                capacity: optional_numeric_value(batch, "capacity", row)?,
                flow: numeric_value_or_default(batch, "flow", row, 0.0)?,
            });
        }
    }
    Ok(edges)
}

pub fn read_demands_batches(batches: &[RecordBatch]) -> Result<Vec<Demand>, String> {
    let mut demands = Vec::new();
    for batch in batches {
        let origins = required_column(batch, "origin_id")?;
        let destinations = required_column(batch, "destination_id")?;
        for row in 0..batch.num_rows() {
            demands.push(Demand {
                origin: integer_value(origins.as_ref(), "origin_id", row)?,
                destination: integer_value(destinations.as_ref(), "destination_id", row)?,
                flow: required_numeric_value(batch, "flow", row)?,
            });
        }
    }
    Ok(demands)
}

pub fn read_od_flows_batches(batches: &[RecordBatch]) -> Result<Vec<OdFlow>, String> {
    let mut od_flows = Vec::new();
    for batch in batches {
        let origins = required_column(batch, "origin_id")?;
        let destinations = required_column(batch, "destination_id")?;
        let edge_paths = batch
            .column_by_name("edge_path")
            .ok_or_else(|| "missing required column edge_path".to_string())?;
        for row in 0..batch.num_rows() {
            od_flows.push(OdFlow {
                origin: integer_value(origins.as_ref(), "origin_id", row)?,
                destination: integer_value(destinations.as_ref(), "destination_id", row)?,
                flow: required_numeric_value(batch, "flow", row)?,
                edge_path: edge_path_value(edge_paths.as_ref(), row)?,
                cost: required_numeric_value(batch, "cost", row)?,
            });
        }
    }
    Ok(od_flows)
}

pub fn allocation_to_ipc(output: &AllocationOutput) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let (od_flows, network_flows, unassigned_od) = allocation_to_batches(output)?;
    Ok((
        write_batch(od_flows)?,
        write_batch(network_flows)?,
        write_batch(unassigned_od)?,
    ))
}

pub fn disruption_to_ipc(
    output: &DisruptionOutput,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let (rerouted_flows, network_flows, isolated_od, losses) = disruption_to_batches(output)?;
    Ok((
        write_batch(rerouted_flows)?,
        write_batch(network_flows)?,
        write_batch(isolated_od)?,
        write_batch(losses)?,
    ))
}

pub fn allocation_to_batches(
    output: &AllocationOutput,
) -> Result<(RecordBatch, RecordBatch, RecordBatch), String> {
    Ok((
        od_flows_batch(&output.od_flows)?,
        network_flows_batch(&output.network_flows)?,
        demands_batch(&output.unassigned_od)?,
    ))
}

pub fn disruption_to_batches(
    output: &DisruptionOutput,
) -> Result<(RecordBatch, RecordBatch, RecordBatch, RecordBatch), String> {
    Ok((
        od_flows_batch(&output.rerouted_flows)?,
        network_flows_batch(&output.network_flows)?,
        demands_batch(&output.isolated_od)?,
        losses_batch(&output.losses)?,
    ))
}

fn read_batches(bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(|error| error.to_string())?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

fn write_batch(batch: RecordBatch) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, batch.schema().as_ref())
            .map_err(|error| error.to_string())?;
        writer.write(&batch).map_err(|error| error.to_string())?;
        writer.finish().map_err(|error| error.to_string())?;
    }
    Ok(buffer)
}

fn od_flows_batch(rows: &[OdFlow]) -> Result<RecordBatch, String> {
    let mut path_builder = ListBuilder::new(UInt64Builder::new());
    for row in rows {
        for edge_id in &row.edge_path {
            path_builder.values().append_value(*edge_id as u64);
        }
        path_builder.append(true);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", DataType::UInt64, false),
        Field::new("destination_id", DataType::UInt64, false),
        Field::new("flow", DataType::Float64, false),
        Field::new(
            "edge_path",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            false,
        ),
        Field::new("cost", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            integer_array(rows.iter().map(|row| row.origin)),
            integer_array(rows.iter().map(|row| row.destination)),
            float_array(rows.iter().map(|row| row.flow)),
            Arc::new(path_builder.finish()) as ArrayRef,
            float_array(rows.iter().map(|row| row.cost)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn network_flows_batch(rows: &[EdgeFlow]) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::UInt64, false),
        Field::new("edge_from", DataType::UInt64, false),
        Field::new("edge_to", DataType::UInt64, false),
        Field::new("cost", DataType::Float64, false),
        Field::new("capacity", DataType::Float64, true),
        Field::new("flow", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            integer_array(rows.iter().map(|row| row.edge_id)),
            integer_array(rows.iter().map(|row| row.edge_from)),
            integer_array(rows.iter().map(|row| row.edge_to)),
            float_array(rows.iter().map(|row| row.cost)),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.capacity).collect::<Vec<_>>(),
            )) as ArrayRef,
            float_array(rows.iter().map(|row| row.flow)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn demands_batch(rows: &[Demand]) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", DataType::UInt64, false),
        Field::new("destination_id", DataType::UInt64, false),
        Field::new("flow", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            integer_array(rows.iter().map(|row| row.origin)),
            integer_array(rows.iter().map(|row| row.destination)),
            float_array(rows.iter().map(|row| row.flow)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn losses_batch(rows: &[Loss]) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", DataType::UInt64, false),
        Field::new("destination_id", DataType::UInt64, false),
        Field::new("flow", DataType::Float64, false),
        Field::new("initial_cost", DataType::Float64, false),
        Field::new("disrupted_cost", DataType::Float64, false),
        Field::new("rerouting_loss", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            integer_array(rows.iter().map(|row| row.origin)),
            integer_array(rows.iter().map(|row| row.destination)),
            float_array(rows.iter().map(|row| row.flow)),
            float_array(rows.iter().map(|row| row.initial_cost)),
            float_array(rows.iter().map(|row| row.disrupted_cost)),
            float_array(rows.iter().map(|row| row.rerouting_loss)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn required_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing required column {name}"))
}

fn integer_value(array: &dyn Array, name: &str, row: usize) -> Result<usize, String> {
    if array.is_null(row) {
        return Err(format!("column {name} cannot contain null values"));
    }
    if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} exceeds usize"))
    } else if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
        Ok(array.value(row) as usize)
    } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else {
        Err(format!("column {name} must be an integer id"))
    }
}

fn required_numeric_value(batch: &RecordBatch, name: &str, row: usize) -> Result<f64, String> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing required column {name}"))?;
    numeric_value(array.as_ref(), name, row)
}

fn numeric_value_or_default(
    batch: &RecordBatch,
    name: &str,
    row: usize,
    default: f64,
) -> Result<f64, String> {
    let Some(array) = batch.column_by_name(name) else {
        return Ok(default);
    };
    if array.is_null(row) {
        return Ok(default);
    }
    numeric_value(array.as_ref(), name, row)
}

fn optional_numeric_value(
    batch: &RecordBatch,
    name: &str,
    row: usize,
) -> Result<Option<f64>, String> {
    let Some(array) = batch.column_by_name(name) else {
        return Ok(None);
    };
    if array.is_null(row) {
        return Ok(None);
    }
    numeric_value(array.as_ref(), name, row).map(Some)
}

fn numeric_value(array: &dyn Array, name: &str, row: usize) -> Result<f64, String> {
    if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        Ok(array.value(row))
    } else if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        Ok(array.value(row) as f64)
    } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        Ok(array.value(row) as f64)
    } else if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        Ok(array.value(row) as f64)
    } else if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
        Ok(array.value(row) as f64)
    } else if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
        Ok(array.value(row) as f64)
    } else {
        Err(format!("column {name} must be numeric"))
    }
}

fn edge_path_value(array: &dyn Array, row: usize) -> Result<Vec<usize>, String> {
    if let Some(paths) = array.as_any().downcast_ref::<ListArray>() {
        let values = paths.value(row);
        return edge_path_values(values.as_ref());
    }

    if let Some(paths) = array.as_any().downcast_ref::<LargeListArray>() {
        let values = paths.value(row);
        return edge_path_values(values.as_ref());
    }

    Err("edge_path must be list<uint64> or another integer list".to_string())
}

fn edge_path_values(values: &dyn Array) -> Result<Vec<usize>, String> {
    if let Some(values) = values.as_any().downcast_ref::<UInt64Array>() {
        return Ok((0..values.len())
            .map(|index| {
                usize::try_from(values.value(index))
                    .map_err(|_| "edge_path value exceeds usize".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?);
    }
    if let Some(values) = values.as_any().downcast_ref::<UInt32Array>() {
        return Ok((0..values.len())
            .map(|index| values.value(index) as usize)
            .collect());
    }
    if let Some(values) = values.as_any().downcast_ref::<Int64Array>() {
        return (0..values.len())
            .map(|index| {
                usize::try_from(values.value(index))
                    .map_err(|_| "edge_path values must be non-negative".to_string())
            })
            .collect();
    }
    if let Some(values) = values.as_any().downcast_ref::<Int32Array>() {
        return (0..values.len())
            .map(|index| {
                usize::try_from(values.value(index))
                    .map_err(|_| "edge_path values must be non-negative".to_string())
            })
            .collect();
    }
    Err("edge_path list values must be integer ids".to_string())
}

fn float_array(values: impl Iterator<Item = f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values.collect::<Vec<_>>())) as ArrayRef
}

fn integer_array(values: impl Iterator<Item = usize>) -> ArrayRef {
    Arc::new(UInt64Array::from(
        values.map(|value| value as u64).collect::<Vec<_>>(),
    )) as ArrayRef
}
