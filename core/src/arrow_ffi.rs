use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray, ListArray,
    ListBuilder, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsuleMethods;
use pyo3::types::{PyAny, PyCapsule};

use crate::core::{AllocationOutput, Demand, DisruptionOutput, Edge, EdgeFlow, Loss, OdFlow};

pub struct DisruptionInputs {
    pub affected_flows: Vec<OdFlow>,
    pub current_edge_flows: Vec<f64>,
    pub initial_costs_by_od: Vec<(usize, usize, f64)>,
}

unsafe extern "C" fn drop_arrow_array_stream_capsule(capsule: *mut ffi::PyObject) {
    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, c"arrow_array_stream".as_ptr()) };
    if ptr.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(ptr.cast::<FFI_ArrowArrayStream>()));
    }
}

fn read_batches_from_pyarrow_table(table: &Bound<'_, PyAny>) -> Result<Vec<RecordBatch>, String> {
    let stream_obj = table
        .call_method0("__arrow_c_stream__")
        .map_err(|error| error.to_string())?;
    let capsule = stream_obj
        .cast_into::<PyCapsule>()
        .map_err(|error| error.to_string())?;
    let stream_ptr = capsule
        .pointer_checked(Some(c"arrow_array_stream"))
        .map_err(|error| error.to_string())?;

    let reader = unsafe {
        ArrowArrayStreamReader::from_raw(stream_ptr.as_ptr().cast::<FFI_ArrowArrayStream>())
    }
    .map_err(|error| error.to_string())?;

    let batches = reader
        .collect::<Result<Vec<RecordBatch>, ArrowError>>()
        .map_err(|error| error.to_string())?;
    Ok(batches)
}

fn batches_to_pyarrow_stream<'py>(
    py: Python<'py>,
    batches: Vec<RecordBatch>,
) -> Result<Bound<'py, PyAny>, String> {
    let schema = batches
        .first()
        .map(|batch| batch.schema())
        .ok_or_else(|| "cannot export empty record batch stream".to_string())?;
    let iter = batches.into_iter().map(Ok);
    let reader = Box::new(RecordBatchIterator::new(iter, schema));
    let stream = FFI_ArrowArrayStream::new(reader);

    let stream_ptr = NonNull::new(Box::into_raw(Box::new(stream)).cast::<c_void>())
        .ok_or_else(|| "failed to allocate Arrow stream capsule".to_string())?;
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            stream_ptr,
            c"arrow_array_stream",
            Some(drop_arrow_array_stream_capsule),
        )
    }
    .map_err(|error| error.to_string())?;

    Ok(capsule.into_any())
}

pub fn read_network_ffi(table: &Bound<'_, PyAny>) -> Result<Vec<crate::core::Edge>, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    read_network_batches(&batches)
}

pub fn read_demands_ffi(table: &Bound<'_, PyAny>) -> Result<Vec<crate::core::Demand>, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    read_demands_batches(&batches)
}

pub fn read_disruption_inputs_ffi(
    table: &Bound<'_, PyAny>,
    failed_edges: &[usize],
    initial_edge_capacity: usize,
) -> Result<DisruptionInputs, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    read_disruption_inputs_batches(&batches, failed_edges, initial_edge_capacity)
}

pub fn allocation_to_ffi<'py>(
    py: Python<'py>,
    output: &AllocationOutput,
) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>), String> {
    let (od_flows, network_flows, unassigned_od) = allocation_to_batches(output)?;
    Ok((
        batches_to_pyarrow_stream(py, vec![od_flows])?,
        batches_to_pyarrow_stream(py, vec![network_flows])?,
        batches_to_pyarrow_stream(py, vec![unassigned_od])?,
    ))
}

pub fn disruption_to_ffi<'py>(
    py: Python<'py>,
    output: &DisruptionOutput,
) -> Result<
    (
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
    ),
    String,
> {
    let (rerouted_flows, network_flows, isolated_od, losses) = disruption_to_batches(output)?;
    Ok((
        batches_to_pyarrow_stream(py, vec![rerouted_flows])?,
        batches_to_pyarrow_stream(py, vec![network_flows])?,
        batches_to_pyarrow_stream(py, vec![isolated_od])?,
        batches_to_pyarrow_stream(py, vec![losses])?,
    ))
}

fn read_disruption_inputs_batches(
    batches: &[RecordBatch],
    failed_edges: &[usize],
    initial_edge_capacity: usize,
) -> Result<DisruptionInputs, String> {
    let mut failed_edge_flags = vec![false; initial_edge_capacity.max(1)];
    for edge_id in failed_edges {
        if *edge_id >= failed_edge_flags.len() {
            failed_edge_flags.resize(*edge_id + 1, false);
        }
        failed_edge_flags[*edge_id] = true;
    }

    let mut current_edge_flows = vec![0.0; initial_edge_capacity];
    let mut affected_flows = Vec::new();
    let mut initial_costs_by_od: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for batch in batches {
        let origins = required_column(batch, "origin_id")?;
        let destinations = required_column(batch, "destination_id")?;
        let flows = required_column(batch, "flow")?;
        let costs = required_column(batch, "cost")?;
        let edge_paths = required_column(batch, "edge_path")?;

        for row in 0..batch.num_rows() {
            let origin = integer_value(origins.as_ref(), "origin_id", row)?;
            let destination = integer_value(destinations.as_ref(), "destination_id", row)?;
            let flow = numeric_value(flows.as_ref(), "flow", row)?;
            let cost = numeric_value(costs.as_ref(), "cost", row)?;
            let edge_path = edge_path_value(edge_paths.as_ref(), row)?;

            let mut is_affected = false;
            for edge_id in &edge_path {
                if *edge_id >= current_edge_flows.len() {
                    current_edge_flows.resize(*edge_id + 1, 0.0);
                }
                current_edge_flows[*edge_id] += flow;

                if *edge_id >= failed_edge_flags.len() {
                    failed_edge_flags.resize(*edge_id + 1, false);
                }
                if failed_edge_flags[*edge_id] {
                    is_affected = true;
                }
            }

            if is_affected {
                *initial_costs_by_od
                    .entry((origin, destination))
                    .or_insert(0.0) += cost;
                affected_flows.push(OdFlow {
                    origin,
                    destination,
                    flow,
                    edge_path,
                    cost,
                });
            }
        }
    }

    let initial_costs_by_od = initial_costs_by_od
        .into_iter()
        .map(|((origin, destination), cost)| (origin, destination, cost))
        .collect::<Vec<_>>();

    Ok(DisruptionInputs {
        affected_flows,
        current_edge_flows,
        initial_costs_by_od,
    })
}

fn required_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow::array::ArrayRef, String> {
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

fn numeric_value(array: &dyn Array, name: &str, row: usize) -> Result<f64, String> {
    if array.is_null(row) {
        return Err(format!("column {name} cannot contain null values"));
    }
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
    if array.is_null(row) {
        return Err("edge_path cannot contain null values".to_string());
    }

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

fn float_array(values: impl Iterator<Item = f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values.collect::<Vec<_>>())) as ArrayRef
}

fn integer_array(values: impl Iterator<Item = usize>) -> ArrayRef {
    Arc::new(UInt64Array::from(
        values.map(|value| value as u64).collect::<Vec<_>>(),
    )) as ArrayRef
}
