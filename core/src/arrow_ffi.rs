use std::ffi::c_void;
use std::ptr::NonNull;

use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsuleMethods;
use pyo3::types::{PyAny, PyCapsule};

use crate::arrow_ipc;
use crate::core::{AllocationOutput, DisruptionOutput};

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

pub fn read_od_flows_ffi(table: &Bound<'_, PyAny>) -> Result<Vec<crate::core::OdFlow>, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    arrow_ipc::read_od_flows_batches(&batches)
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
