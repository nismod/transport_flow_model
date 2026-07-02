use std::ffi::c_void;
use std::ptr::NonNull;

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray, ListArray,
    UInt32Array, UInt64Array,
};
use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsuleMethods;
use pyo3::types::{PyAny, PyCapsule};

use crate::arrow_ipc;
use crate::core::{AllocationOutput, DisruptionOutput, OdFlow};

pub struct DisruptionInputs {
    pub affected_flows: Vec<OdFlow>,
    pub current_edge_flows: Vec<f64>,
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
    arrow_ipc::read_network_batches(&batches)
}

pub fn read_demands_ffi(table: &Bound<'_, PyAny>) -> Result<Vec<crate::core::Demand>, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    arrow_ipc::read_demands_batches(&batches)
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
    let (od_flows, network_flows, unassigned_od) = arrow_ipc::allocation_to_batches(output)?;
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
    let (rerouted_flows, network_flows, isolated_od, losses) =
        arrow_ipc::disruption_to_batches(output)?;
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

    Ok(DisruptionInputs {
        affected_flows,
        current_edge_flows,
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
