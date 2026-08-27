use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int16Builder, Int32Array,
    Int32Builder, Int64Array, Int64Builder, Int8Array, Int8Builder, LargeListArray,
    LargeStringArray, LargeStringBuilder, ListArray, ListBuilder, StringArray, StringBuilder,
    UInt16Array, UInt16Builder, UInt32Array, UInt32Builder, UInt64Array, UInt64Builder, UInt8Array,
    UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsuleMethods;
use pyo3::types::{PyAny, PyBool, PyCapsule};

use crate::core::{AllocationOutput, Demand, DisruptionOutput, Edge, EdgeFlow, Loss, OdFlow};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdKind {
    Utf8,
    LargeUtf8,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl IdKind {
    fn logical_name(self) -> &'static str {
        match self {
            Self::Utf8 | Self::LargeUtf8 => "string",
            Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64 => "signed integer",
            Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64 => "unsigned integer",
        }
    }

    fn is_compatible_with(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Utf8 | Self::LargeUtf8, Self::Utf8 | Self::LargeUtf8)
                | (
                    Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64,
                    Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64
                )
                | (
                    Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64,
                    Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64
                )
        )
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Utf8 => DataType::Utf8,
            Self::LargeUtf8 => DataType::LargeUtf8,
            Self::Int8 => DataType::Int8,
            Self::Int16 => DataType::Int16,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::UInt8 => DataType::UInt8,
            Self::UInt16 => DataType::UInt16,
            Self::UInt32 => DataType::UInt32,
            Self::UInt64 => DataType::UInt64,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExternalId {
    String(String),
    Signed(i64),
    Unsigned(u64),
}

impl Hash for ExternalId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::String(value) => {
                0u8.hash(state);
                value.hash(state);
            }
            Self::Signed(value) => {
                1u8.hash(state);
                value.hash(state);
            }
            Self::Unsigned(value) => {
                2u8.hash(state);
                value.hash(state);
            }
        }
    }
}

enum IdColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Int8(&'a Int8Array),
    Int16(&'a Int16Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt8(&'a UInt8Array),
    UInt16(&'a UInt16Array),
    UInt32(&'a UInt32Array),
    UInt64(&'a UInt64Array),
}

impl<'a> IdColumn<'a> {
    fn new(array: &'a dyn Array, name: &str) -> Result<Self, String> {
        match array.data_type() {
            DataType::Utf8 => Ok(Self::Utf8(
                array.as_any().downcast_ref::<StringArray>().unwrap(),
            )),
            DataType::LargeUtf8 => Ok(Self::LargeUtf8(
                array.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            )),
            DataType::Int8 => Ok(Self::Int8(
                array.as_any().downcast_ref::<Int8Array>().unwrap(),
            )),
            DataType::Int16 => Ok(Self::Int16(
                array.as_any().downcast_ref::<Int16Array>().unwrap(),
            )),
            DataType::Int32 => Ok(Self::Int32(
                array.as_any().downcast_ref::<Int32Array>().unwrap(),
            )),
            DataType::Int64 => Ok(Self::Int64(
                array.as_any().downcast_ref::<Int64Array>().unwrap(),
            )),
            DataType::UInt8 => Ok(Self::UInt8(
                array.as_any().downcast_ref::<UInt8Array>().unwrap(),
            )),
            DataType::UInt16 => Ok(Self::UInt16(
                array.as_any().downcast_ref::<UInt16Array>().unwrap(),
            )),
            DataType::UInt32 => Ok(Self::UInt32(
                array.as_any().downcast_ref::<UInt32Array>().unwrap(),
            )),
            DataType::UInt64 => Ok(Self::UInt64(
                array.as_any().downcast_ref::<UInt64Array>().unwrap(),
            )),
            _ => Err(format!(
                "column {name} has unsupported id type {}; supported id types are string and integer",
                array.data_type()
            )),
        }
    }

    fn kind(&self) -> IdKind {
        match self {
            Self::Utf8(_) => IdKind::Utf8,
            Self::LargeUtf8(_) => IdKind::LargeUtf8,
            Self::Int8(_) => IdKind::Int8,
            Self::Int16(_) => IdKind::Int16,
            Self::Int32(_) => IdKind::Int32,
            Self::Int64(_) => IdKind::Int64,
            Self::UInt8(_) => IdKind::UInt8,
            Self::UInt16(_) => IdKind::UInt16,
            Self::UInt32(_) => IdKind::UInt32,
            Self::UInt64(_) => IdKind::UInt64,
        }
    }

    fn value(&self, name: &str, row: usize) -> Result<ExternalId, String> {
        if self.is_null(row) {
            return Err(format!("column {name} cannot contain null id values"));
        }
        Ok(match self {
            Self::Utf8(array) => ExternalId::String(array.value(row).to_string()),
            Self::LargeUtf8(array) => ExternalId::String(array.value(row).to_string()),
            Self::Int8(array) => ExternalId::Signed(array.value(row) as i64),
            Self::Int16(array) => ExternalId::Signed(array.value(row) as i64),
            Self::Int32(array) => ExternalId::Signed(array.value(row) as i64),
            Self::Int64(array) => ExternalId::Signed(array.value(row)),
            Self::UInt8(array) => ExternalId::Unsigned(array.value(row) as u64),
            Self::UInt16(array) => ExternalId::Unsigned(array.value(row) as u64),
            Self::UInt32(array) => ExternalId::Unsigned(array.value(row) as u64),
            Self::UInt64(array) => ExternalId::Unsigned(array.value(row)),
        })
    }

    fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Utf8(array) => array.is_null(row),
            Self::LargeUtf8(array) => array.is_null(row),
            Self::Int8(array) => array.is_null(row),
            Self::Int16(array) => array.is_null(row),
            Self::Int32(array) => array.is_null(row),
            Self::Int64(array) => array.is_null(row),
            Self::UInt8(array) => array.is_null(row),
            Self::UInt16(array) => array.is_null(row),
            Self::UInt32(array) => array.is_null(row),
            Self::UInt64(array) => array.is_null(row),
        }
    }
}

/// A numeric column resolved and downcast once per batch.
///
/// Mirrors [`IdColumn`]: the row loops that build `Edge`, `Demand` and
/// `OdFlow` values used to call `column_by_name` and walk a `downcast_ref`
/// chain for every cell, which dominated the cost of parsing a network.
enum NumericColumn<'a> {
    Float64(&'a Float64Array),
    Float32(&'a Float32Array),
    Int8(&'a Int8Array),
    Int16(&'a Int16Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt8(&'a UInt8Array),
    UInt16(&'a UInt16Array),
    UInt32(&'a UInt32Array),
    UInt64(&'a UInt64Array),
}

impl<'a> NumericColumn<'a> {
    fn new(array: &'a dyn Array, name: &str) -> Result<Self, String> {
        match array.data_type() {
            DataType::Float64 => Ok(Self::Float64(
                array.as_any().downcast_ref::<Float64Array>().unwrap(),
            )),
            DataType::Float32 => Ok(Self::Float32(
                array.as_any().downcast_ref::<Float32Array>().unwrap(),
            )),
            DataType::Int8 => Ok(Self::Int8(
                array.as_any().downcast_ref::<Int8Array>().unwrap(),
            )),
            DataType::Int16 => Ok(Self::Int16(
                array.as_any().downcast_ref::<Int16Array>().unwrap(),
            )),
            DataType::Int32 => Ok(Self::Int32(
                array.as_any().downcast_ref::<Int32Array>().unwrap(),
            )),
            DataType::Int64 => Ok(Self::Int64(
                array.as_any().downcast_ref::<Int64Array>().unwrap(),
            )),
            DataType::UInt8 => Ok(Self::UInt8(
                array.as_any().downcast_ref::<UInt8Array>().unwrap(),
            )),
            DataType::UInt16 => Ok(Self::UInt16(
                array.as_any().downcast_ref::<UInt16Array>().unwrap(),
            )),
            DataType::UInt32 => Ok(Self::UInt32(
                array.as_any().downcast_ref::<UInt32Array>().unwrap(),
            )),
            DataType::UInt64 => Ok(Self::UInt64(
                array.as_any().downcast_ref::<UInt64Array>().unwrap(),
            )),
            _ => Err(format!("column {name} must be numeric")),
        }
    }

    /// Resolve `name` in `batch`, or `None` when the column is absent.
    fn optional(batch: &'a RecordBatch, name: &str) -> Result<Option<Self>, String> {
        match batch.column_by_name(name) {
            Some(array) => Self::new(array.as_ref(), name).map(Some),
            None => Ok(None),
        }
    }

    /// Resolve `name` in `batch`, erroring when the column is absent.
    fn required(batch: &'a RecordBatch, name: &str) -> Result<Self, String> {
        Self::new(required_column(batch, name)?.as_ref(), name)
    }

    fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Float64(array) => array.is_null(row),
            Self::Float32(array) => array.is_null(row),
            Self::Int8(array) => array.is_null(row),
            Self::Int16(array) => array.is_null(row),
            Self::Int32(array) => array.is_null(row),
            Self::Int64(array) => array.is_null(row),
            Self::UInt8(array) => array.is_null(row),
            Self::UInt16(array) => array.is_null(row),
            Self::UInt32(array) => array.is_null(row),
            Self::UInt64(array) => array.is_null(row),
        }
    }

    fn value_unchecked(&self, row: usize) -> f64 {
        match self {
            Self::Float64(array) => array.value(row),
            Self::Float32(array) => array.value(row) as f64,
            Self::Int8(array) => array.value(row) as f64,
            Self::Int16(array) => array.value(row) as f64,
            Self::Int32(array) => array.value(row) as f64,
            Self::Int64(array) => array.value(row) as f64,
            Self::UInt8(array) => array.value(row) as f64,
            Self::UInt16(array) => array.value(row) as f64,
            Self::UInt32(array) => array.value(row) as f64,
            Self::UInt64(array) => array.value(row) as f64,
        }
    }

    /// Value at `row`; nulls are an error.
    fn value(&self, name: &str, row: usize) -> Result<f64, String> {
        if self.is_null(row) {
            return Err(format!("column {name} cannot contain null values"));
        }
        Ok(self.value_unchecked(row))
    }
}

/// Value at `row` from an optional column, falling back to `default` when the
/// column is absent or the value is null.
fn numeric_or_default(column: Option<&NumericColumn>, row: usize, default: f64) -> f64 {
    match column {
        Some(column) if !column.is_null(row) => column.value_unchecked(row),
        _ => default,
    }
}

/// Value at `row` from an optional column, `None` when absent or null.
fn optional_numeric(column: Option<&NumericColumn>, row: usize) -> Option<f64> {
    match column {
        Some(column) if !column.is_null(row) => Some(column.value_unchecked(row)),
        _ => None,
    }
}

pub struct PreparedNetwork {
    pub edges: Vec<Edge>,
    edge_ids: Vec<ExternalId>,
    node_ids: Vec<ExternalId>,
    edge_id_to_internal: HashMap<ExternalId, usize>,
    node_id_to_internal: HashMap<ExternalId, usize>,
    edge_kind: IdKind,
    node_kind: IdKind,
}

pub struct PreparedDemands {
    pub demands: Vec<Demand>,
}

pub struct PreparedOdFlows {
    pub inputs: DisruptionInputs,
    pub failed_edges: Vec<usize>,
}

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

pub fn read_network_ffi(table: &Bound<'_, PyAny>) -> Result<Vec<Edge>, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    read_network_batches(&batches)
}

pub fn prepare_network_ffi(table: &Bound<'_, PyAny>) -> Result<PreparedNetwork, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    prepare_network_batches(&batches)
}

pub fn prepare_demands_ffi(
    table: &Bound<'_, PyAny>,
    network: &mut PreparedNetwork,
) -> Result<PreparedDemands, String> {
    let batches = read_batches_from_pyarrow_table(table)?;
    prepare_demands_batches(&batches, network)
}

pub fn prepare_od_flows_ffi(
    table: &Bound<'_, PyAny>,
    failed_edges: &Bound<'_, PyAny>,
    network: &mut PreparedNetwork,
) -> Result<PreparedOdFlows, String> {
    let failed_edges = decode_failed_edges(failed_edges, network)?;
    let batches = read_batches_from_pyarrow_table(table)?;
    let inputs = prepare_disruption_inputs_batches(&batches, &failed_edges, network)?;
    Ok(PreparedOdFlows {
        inputs,
        failed_edges,
    })
}

pub fn allocation_to_ffi<'py>(
    py: Python<'py>,
    output: &AllocationOutput,
    network: &PreparedNetwork,
) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>), String> {
    let (od_flows, network_flows, unassigned_od) = allocation_to_batches(output, network)?;
    Ok((
        batches_to_pyarrow_stream(py, vec![od_flows])?,
        batches_to_pyarrow_stream(py, vec![network_flows])?,
        batches_to_pyarrow_stream(py, vec![unassigned_od])?,
    ))
}

pub fn disruption_to_ffi<'py>(
    py: Python<'py>,
    output: &DisruptionOutput,
    network: &PreparedNetwork,
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
        disruption_to_batches(output, network)?;
    Ok((
        batches_to_pyarrow_stream(py, vec![rerouted_flows])?,
        batches_to_pyarrow_stream(py, vec![network_flows])?,
        batches_to_pyarrow_stream(py, vec![isolated_od])?,
        batches_to_pyarrow_stream(py, vec![losses])?,
    ))
}

pub fn prepare_network_batches(batches: &[RecordBatch]) -> Result<PreparedNetwork, String> {
    let node_kind = first_id_kind(batches, "edge_from")?;
    let edge_kind = first_id_kind(batches, "edge_id")?;
    let mut node_ids = Vec::new();
    let mut node_id_to_internal = HashMap::new();
    let mut edge_ids = Vec::new();
    let mut edge_id_to_internal = HashMap::new();
    let mut edges = Vec::new();

    for batch in batches {
        let from = id_column(batch, "edge_from")?;
        let to = id_column(batch, "edge_to")?;
        let id = id_column(batch, "edge_id")?;
        ensure_compatible_id_kind("edge_from", node_kind, "edge_from", from.kind())?;
        ensure_compatible_id_kind("edge_to", node_kind, "edge_to", to.kind())?;
        ensure_compatible_id_kind("edge_id", edge_kind, "edge_id", id.kind())?;
        let cost = NumericColumn::optional(batch, "cost")?;
        let capacity = NumericColumn::optional(batch, "capacity")?;
        let flow = NumericColumn::optional(batch, "flow")?;

        for row in 0..batch.num_rows() {
            let edge_external_id = id.value("edge_id", row)?;
            if edge_id_to_internal.contains_key(&edge_external_id) {
                return Err(format!("duplicate edge_id value at row {row}"));
            }
            let edge_id = edge_ids.len();
            edge_id_to_internal.insert(edge_external_id.clone(), edge_id);
            edge_ids.push(edge_external_id);

            let from = intern_node(
                from.value("edge_from", row)?,
                &mut node_ids,
                &mut node_id_to_internal,
            );
            let to = intern_node(
                to.value("edge_to", row)?,
                &mut node_ids,
                &mut node_id_to_internal,
            );

            edges.push(Edge {
                from,
                to,
                id: edge_id,
                cost: numeric_or_default(cost.as_ref(), row, 1.0),
                capacity: optional_numeric(capacity.as_ref(), row),
                flow: numeric_or_default(flow.as_ref(), row, 0.0),
            });
        }
    }

    Ok(PreparedNetwork {
        edges,
        edge_ids,
        node_ids,
        edge_id_to_internal,
        node_id_to_internal,
        edge_kind,
        node_kind,
    })
}

fn prepare_demands_batches(
    batches: &[RecordBatch],
    network: &mut PreparedNetwork,
) -> Result<PreparedDemands, String> {
    let mut demands = Vec::new();
    for batch in batches {
        let origins = id_column(batch, "origin_id")?;
        let destinations = id_column(batch, "destination_id")?;
        ensure_compatible_id_kind("origin_id", network.node_kind, "edge_from", origins.kind())?;
        ensure_compatible_id_kind(
            "destination_id",
            network.node_kind,
            "edge_from",
            destinations.kind(),
        )?;
        let flows = NumericColumn::required(batch, "flow")?;
        for row in 0..batch.num_rows() {
            let origin = network.intern_node_id(origins.value("origin_id", row)?);
            let destination = network.intern_node_id(destinations.value("destination_id", row)?);
            demands.push(Demand {
                origin,
                destination,
                flow: flows.value("flow", row)?,
            });
        }
    }
    Ok(PreparedDemands { demands })
}

fn prepare_disruption_inputs_batches(
    batches: &[RecordBatch],
    failed_edges: &[usize],
    network: &mut PreparedNetwork,
) -> Result<DisruptionInputs, String> {
    let mut failed_edge_flags = vec![false; network.edge_ids.len().max(1)];
    for edge_id in failed_edges {
        failed_edge_flags[*edge_id] = true;
    }

    let mut current_edge_flows = vec![0.0; network.edge_ids.len()];
    let mut affected_flows = Vec::new();
    let mut initial_costs_by_od: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for batch in batches {
        let origins = id_column(batch, "origin_id")?;
        let destinations = id_column(batch, "destination_id")?;
        ensure_compatible_id_kind("origin_id", network.node_kind, "edge_from", origins.kind())?;
        ensure_compatible_id_kind(
            "destination_id",
            network.node_kind,
            "edge_from",
            destinations.kind(),
        )?;
        let edge_paths = required_column(batch, "edge_path")?;
        ensure_edge_path_kind(edge_paths.as_ref(), network.edge_kind)?;
        let flows = NumericColumn::required(batch, "flow")?;
        let costs = NumericColumn::required(batch, "cost")?;

        for row in 0..batch.num_rows() {
            let origin = network.intern_node_id(origins.value("origin_id", row)?);
            let destination = network.intern_node_id(destinations.value("destination_id", row)?);
            let flow = flows.value("flow", row)?;
            let cost = costs.value("cost", row)?;
            let edge_path = external_edge_path_value(edge_paths.as_ref(), row, network)?;

            let mut is_affected = false;
            for edge_id in &edge_path {
                current_edge_flows[*edge_id] += flow;
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

impl PreparedNetwork {
    fn intern_node_id(&mut self, value: ExternalId) -> usize {
        intern_node(value, &mut self.node_ids, &mut self.node_id_to_internal)
    }
}

fn intern_node(
    value: ExternalId,
    node_ids: &mut Vec<ExternalId>,
    node_id_to_internal: &mut HashMap<ExternalId, usize>,
) -> usize {
    if let Some(node_id) = node_id_to_internal.get(&value) {
        *node_id
    } else {
        let node_id = node_ids.len();
        node_id_to_internal.insert(value.clone(), node_id);
        node_ids.push(value);
        node_id
    }
}

fn decode_failed_edges(
    failed_edges: &Bound<'_, PyAny>,
    network: &PreparedNetwork,
) -> Result<Vec<usize>, String> {
    let iterator = failed_edges
        .try_iter()
        .map_err(|_| "failed_edges must be an iterable of edge ids".to_string())?;
    let mut decoded = Vec::new();
    for value in iterator {
        let value = value.map_err(|error| error.to_string())?;
        let external_id = py_external_id(&value, network.edge_kind)?;
        if let Some(edge_id) = network.edge_id_to_internal.get(&external_id) {
            decoded.push(*edge_id);
        }
    }
    Ok(decoded)
}

fn py_external_id(value: &Bound<'_, PyAny>, kind: IdKind) -> Result<ExternalId, String> {
    if value.is_instance_of::<PyBool>() {
        return Err("failed_edges values must match the network edge_id type".to_string());
    }
    match kind {
        IdKind::Utf8 | IdKind::LargeUtf8 => value
            .extract::<String>()
            .map(ExternalId::String)
            .map_err(|_| "failed_edges values must be strings".to_string()),
        IdKind::Int8 | IdKind::Int16 | IdKind::Int32 | IdKind::Int64 => value
            .extract::<i64>()
            .map(ExternalId::Signed)
            .map_err(|_| "failed_edges values must be signed integers".to_string()),
        IdKind::UInt8 | IdKind::UInt16 | IdKind::UInt32 | IdKind::UInt64 => value
            .extract::<u64>()
            .map(ExternalId::Unsigned)
            .map_err(|_| "failed_edges values must be unsigned integers".to_string()),
    }
}

fn first_id_kind(batches: &[RecordBatch], name: &str) -> Result<IdKind, String> {
    let batch = batches
        .first()
        .ok_or_else(|| "input table must contain at least one record batch".to_string())?;
    id_column(batch, name).map(|column| column.kind())
}

fn id_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<IdColumn<'a>, String> {
    let array = required_column(batch, name)?;
    IdColumn::new(array.as_ref(), name)
}

fn ensure_compatible_id_kind(
    name: &str,
    expected: IdKind,
    expected_name: &str,
    actual: IdKind,
) -> Result<(), String> {
    if expected.is_compatible_with(actual) {
        Ok(())
    } else {
        Err(format!(
            "column {name} id type ({}) must match column {expected_name} id type ({})",
            actual.logical_name(),
            expected.logical_name()
        ))
    }
}

fn ensure_edge_path_kind(array: &dyn Array, edge_kind: IdKind) -> Result<(), String> {
    let value_type = match array.data_type() {
        DataType::List(field) | DataType::LargeList(field) => field.data_type(),
        _ => {
            return Err("edge_path must be a list of edge ids".to_string());
        }
    };
    let values_kind = id_kind_from_data_type(value_type).ok_or_else(|| {
        "edge_path list values have unsupported id type; supported id types are string and integer"
            .to_string()
    })?;
    ensure_compatible_id_kind("edge_path", edge_kind, "edge_id", values_kind)
}

fn id_kind_from_data_type(data_type: &DataType) -> Option<IdKind> {
    match data_type {
        DataType::Utf8 => Some(IdKind::Utf8),
        DataType::LargeUtf8 => Some(IdKind::LargeUtf8),
        DataType::Int8 => Some(IdKind::Int8),
        DataType::Int16 => Some(IdKind::Int16),
        DataType::Int32 => Some(IdKind::Int32),
        DataType::Int64 => Some(IdKind::Int64),
        DataType::UInt8 => Some(IdKind::UInt8),
        DataType::UInt16 => Some(IdKind::UInt16),
        DataType::UInt32 => Some(IdKind::UInt32),
        DataType::UInt64 => Some(IdKind::UInt64),
        _ => None,
    }
}

fn external_edge_path_value(
    array: &dyn Array,
    row: usize,
    network: &PreparedNetwork,
) -> Result<Vec<usize>, String> {
    if array.is_null(row) {
        return Err("edge_path cannot contain null values".to_string());
    }

    let values = if let Some(paths) = array.as_any().downcast_ref::<ListArray>() {
        paths.value(row)
    } else if let Some(paths) = array.as_any().downcast_ref::<LargeListArray>() {
        paths.value(row)
    } else {
        return Err("edge_path must be a list of edge ids".to_string());
    };

    let values = IdColumn::new(values.as_ref(), "edge_path")?;
    (0..values_len(&values))
        .map(|index| {
            let external_id = values.value("edge_path", index)?;
            network
                .edge_id_to_internal
                .get(&external_id)
                .copied()
                .ok_or_else(|| "edge_path contains an unknown edge_id".to_string())
        })
        .collect()
}

fn values_len(values: &IdColumn<'_>) -> usize {
    match values {
        IdColumn::Utf8(array) => array.len(),
        IdColumn::LargeUtf8(array) => array.len(),
        IdColumn::Int8(array) => array.len(),
        IdColumn::Int16(array) => array.len(),
        IdColumn::Int32(array) => array.len(),
        IdColumn::Int64(array) => array.len(),
        IdColumn::UInt8(array) => array.len(),
        IdColumn::UInt16(array) => array.len(),
        IdColumn::UInt32(array) => array.len(),
        IdColumn::UInt64(array) => array.len(),
    }
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
    } else if let Some(array) = array.as_any().downcast_ref::<UInt16Array>() {
        Ok(array.value(row) as usize)
    } else if let Some(array) = array.as_any().downcast_ref::<UInt8Array>() {
        Ok(array.value(row) as usize)
    } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else if let Some(array) = array.as_any().downcast_ref::<Int16Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else if let Some(array) = array.as_any().downcast_ref::<Int8Array>() {
        usize::try_from(array.value(row)).map_err(|_| format!("column {name} must be non-negative"))
    } else {
        Err(format!("column {name} must be an integer id"))
    }
}

pub fn read_network_batches(batches: &[RecordBatch]) -> Result<Vec<Edge>, String> {
    let mut edges = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    for batch in batches {
        let from = required_column(batch, "edge_from")?;
        let to = required_column(batch, "edge_to")?;
        let id = required_column(batch, "edge_id")?;
        let cost = NumericColumn::optional(batch, "cost")?;
        let capacity = NumericColumn::optional(batch, "capacity")?;
        let flow = NumericColumn::optional(batch, "flow")?;
        for row in 0..batch.num_rows() {
            edges.push(Edge {
                from: integer_value(from.as_ref(), "edge_from", row)?,
                to: integer_value(to.as_ref(), "edge_to", row)?,
                id: integer_value(id.as_ref(), "edge_id", row)?,
                cost: numeric_or_default(cost.as_ref(), row, 1.0),
                capacity: optional_numeric(capacity.as_ref(), row),
                flow: numeric_or_default(flow.as_ref(), row, 0.0),
            });
        }
    }
    Ok(edges)
}

fn allocation_to_batches(
    output: &AllocationOutput,
    network: &PreparedNetwork,
) -> Result<(RecordBatch, RecordBatch, RecordBatch), String> {
    Ok((
        od_flows_batch(&output.od_flows, network)?,
        network_flows_batch(&output.network_flows, network)?,
        demands_batch(&output.unassigned_od, network)?,
    ))
}

fn disruption_to_batches(
    output: &DisruptionOutput,
    network: &PreparedNetwork,
) -> Result<(RecordBatch, RecordBatch, RecordBatch, RecordBatch), String> {
    Ok((
        od_flows_batch(&output.rerouted_flows, network)?,
        network_flows_batch(&output.network_flows, network)?,
        demands_batch(&output.isolated_od, network)?,
        losses_batch(&output.losses, network)?,
    ))
}

fn od_flows_batch(rows: &[OdFlow], network: &PreparedNetwork) -> Result<RecordBatch, String> {
    let edge_path = edge_path_array(rows, network)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", network.node_kind.data_type(), false),
        Field::new("destination_id", network.node_kind.data_type(), false),
        Field::new("flow", DataType::Float64, false),
        Field::new(
            "edge_path",
            DataType::List(Arc::new(Field::new(
                "item",
                network.edge_kind.data_type(),
                true,
            ))),
            false,
        ),
        Field::new("cost", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            external_id_array(
                rows.iter().map(|row| external_node_id(network, row.origin)),
                network.node_kind,
            )?,
            external_id_array(
                rows.iter()
                    .map(|row| external_node_id(network, row.destination)),
                network.node_kind,
            )?,
            float_array(rows.iter().map(|row| row.flow)),
            edge_path,
            float_array(rows.iter().map(|row| row.cost)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn network_flows_batch(
    rows: &[EdgeFlow],
    network: &PreparedNetwork,
) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", network.edge_kind.data_type(), false),
        Field::new("edge_from", network.node_kind.data_type(), false),
        Field::new("edge_to", network.node_kind.data_type(), false),
        Field::new("cost", DataType::Float64, false),
        Field::new("capacity", DataType::Float64, true),
        Field::new("flow", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            external_id_array(
                rows.iter()
                    .map(|row| external_edge_id(network, row.edge_id)),
                network.edge_kind,
            )?,
            external_id_array(
                rows.iter()
                    .map(|row| external_node_id(network, row.edge_from)),
                network.node_kind,
            )?,
            external_id_array(
                rows.iter()
                    .map(|row| external_node_id(network, row.edge_to)),
                network.node_kind,
            )?,
            float_array(rows.iter().map(|row| row.cost)),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.capacity).collect::<Vec<_>>(),
            )) as ArrayRef,
            float_array(rows.iter().map(|row| row.flow)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn demands_batch(rows: &[Demand], network: &PreparedNetwork) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", network.node_kind.data_type(), false),
        Field::new("destination_id", network.node_kind.data_type(), false),
        Field::new("flow", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            external_id_array(
                rows.iter().map(|row| external_node_id(network, row.origin)),
                network.node_kind,
            )?,
            external_id_array(
                rows.iter()
                    .map(|row| external_node_id(network, row.destination)),
                network.node_kind,
            )?,
            float_array(rows.iter().map(|row| row.flow)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn losses_batch(rows: &[Loss], network: &PreparedNetwork) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("origin_id", network.node_kind.data_type(), false),
        Field::new("destination_id", network.node_kind.data_type(), false),
        Field::new("flow", DataType::Float64, false),
        Field::new("initial_cost", DataType::Float64, false),
        Field::new("disrupted_cost", DataType::Float64, false),
        Field::new("rerouting_loss", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            external_id_array(
                rows.iter().map(|row| external_node_id(network, row.origin)),
                network.node_kind,
            )?,
            external_id_array(
                rows.iter()
                    .map(|row| external_node_id(network, row.destination)),
                network.node_kind,
            )?,
            float_array(rows.iter().map(|row| row.flow)),
            float_array(rows.iter().map(|row| row.initial_cost)),
            float_array(rows.iter().map(|row| row.disrupted_cost)),
            float_array(rows.iter().map(|row| row.rerouting_loss)),
        ],
    )
    .map_err(|error| error.to_string())
}

fn external_node_id(network: &PreparedNetwork, id: usize) -> Result<&ExternalId, String> {
    network
        .node_ids
        .get(id)
        .ok_or_else(|| format!("internal node id {id} has no external id"))
}

fn external_edge_id(network: &PreparedNetwork, id: usize) -> Result<&ExternalId, String> {
    network
        .edge_ids
        .get(id)
        .ok_or_else(|| format!("internal edge id {id} has no external id"))
}

fn edge_path_array(rows: &[OdFlow], network: &PreparedNetwork) -> Result<ArrayRef, String> {
    match network.edge_kind {
        IdKind::Utf8 => {
            let mut builder = ListBuilder::new(StringBuilder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_string_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::LargeUtf8 => {
            let mut builder = ListBuilder::new(LargeStringBuilder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_large_string_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int8 => {
            let mut builder = ListBuilder::new(Int8Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_i8_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int16 => {
            let mut builder = ListBuilder::new(Int16Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_i16_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int32 => {
            let mut builder = ListBuilder::new(Int32Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_i32_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int64 => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_i64_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt8 => {
            let mut builder = ListBuilder::new(UInt8Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_u8_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt16 => {
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_u16_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt32 => {
            let mut builder = ListBuilder::new(UInt32Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_u32_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt64 => {
            let mut builder = ListBuilder::new(UInt64Builder::new());
            for row in rows {
                for edge_id in &row.edge_path {
                    append_u64_id(builder.values(), external_edge_id(network, *edge_id)?)?;
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
    }
}

fn external_id_array<'a>(
    values: impl Iterator<Item = Result<&'a ExternalId, String>>,
    kind: IdKind,
) -> Result<ArrayRef, String> {
    match kind {
        IdKind::Utf8 => {
            let mut builder = StringBuilder::new();
            for value in values {
                append_string_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::LargeUtf8 => {
            let mut builder = LargeStringBuilder::new();
            for value in values {
                append_large_string_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int8 => {
            let mut builder = Int8Builder::new();
            for value in values {
                append_i8_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int16 => {
            let mut builder = Int16Builder::new();
            for value in values {
                append_i16_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int32 => {
            let mut builder = Int32Builder::new();
            for value in values {
                append_i32_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::Int64 => {
            let mut builder = Int64Builder::new();
            for value in values {
                append_i64_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt8 => {
            let mut builder = UInt8Builder::new();
            for value in values {
                append_u8_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt16 => {
            let mut builder = UInt16Builder::new();
            for value in values {
                append_u16_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt32 => {
            let mut builder = UInt32Builder::new();
            for value in values {
                append_u32_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        IdKind::UInt64 => {
            let mut builder = UInt64Builder::new();
            for value in values {
                append_u64_id(&mut builder, value?)?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
    }
}

fn append_string_id(builder: &mut StringBuilder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::String(value) => {
            builder.append_value(value);
            Ok(())
        }
        _ => Err("expected string id".to_string()),
    }
}

fn append_large_string_id(
    builder: &mut LargeStringBuilder,
    value: &ExternalId,
) -> Result<(), String> {
    match value {
        ExternalId::String(value) => {
            builder.append_value(value);
            Ok(())
        }
        _ => Err("expected string id".to_string()),
    }
}

fn append_i8_id(builder: &mut Int8Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Signed(value) => i8::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds int8".to_string()),
        _ => Err("expected signed integer id".to_string()),
    }
}

fn append_i16_id(builder: &mut Int16Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Signed(value) => i16::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds int16".to_string()),
        _ => Err("expected signed integer id".to_string()),
    }
}

fn append_i32_id(builder: &mut Int32Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Signed(value) => i32::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds int32".to_string()),
        _ => Err("expected signed integer id".to_string()),
    }
}

fn append_i64_id(builder: &mut Int64Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Signed(value) => {
            builder.append_value(*value);
            Ok(())
        }
        _ => Err("expected signed integer id".to_string()),
    }
}

fn append_u8_id(builder: &mut UInt8Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Unsigned(value) => u8::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds uint8".to_string()),
        _ => Err("expected unsigned integer id".to_string()),
    }
}

fn append_u16_id(builder: &mut UInt16Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Unsigned(value) => u16::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds uint16".to_string()),
        _ => Err("expected unsigned integer id".to_string()),
    }
}

fn append_u32_id(builder: &mut UInt32Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Unsigned(value) => u32::try_from(*value)
            .map(|value| builder.append_value(value))
            .map_err(|_| "external id exceeds uint32".to_string()),
        _ => Err("expected unsigned integer id".to_string()),
    }
}

fn append_u64_id(builder: &mut UInt64Builder, value: &ExternalId) -> Result<(), String> {
    match value {
        ExternalId::Unsigned(value) => {
            builder.append_value(*value);
            Ok(())
        }
        _ => Err("expected unsigned integer id".to_string()),
    }
}

fn float_array(values: impl Iterator<Item = f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values.collect::<Vec<_>>())) as ArrayRef
}

fn integer_array(values: impl Iterator<Item = usize>) -> ArrayRef {
    Arc::new(UInt64Array::from(
        values.map(|value| value as u64).collect::<Vec<_>>(),
    )) as ArrayRef
}

pub fn shortest_paths_to_ffi<'py>(
    py: Python<'py>,
    paths: &[(usize, f64)],
) -> Result<Bound<'py, PyAny>, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::UInt64, false),
        Field::new("cost", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            integer_array(paths.iter().map(|(node_id, _)| *node_id)),
            float_array(paths.iter().map(|(_, cost)| *cost)),
        ],
    )
    .map_err(|error| error.to_string())?;
    batches_to_pyarrow_stream(py, vec![batch])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string_network_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("edge_from", DataType::Utf8, false),
            Field::new("edge_to", DataType::Utf8, false),
            Field::new("edge_id", DataType::Utf8, false),
            Field::new("cost", DataType::Float64, false),
            Field::new("capacity", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "C"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["B", "C", "B"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["AB", "AC", "CB"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 3.0, 5.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![100.0, 100.0, 100.0])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn string_demands_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("origin_id", DataType::Utf8, false),
            Field::new("destination_id", DataType::Utf8, false),
            Field::new("flow", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["B"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![7.0])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    #[test]
    fn prepares_string_network_and_demands() {
        let mut network = prepare_network_batches(&[string_network_batch()]).unwrap();
        let demands = prepare_demands_batches(&[string_demands_batch()], &mut network).unwrap();

        assert_eq!(network.edges[0].from, 0);
        assert_eq!(network.edges[0].to, 1);
        assert_eq!(network.edges[1].from, 0);
        assert_eq!(network.edges[1].to, 2);
        assert_eq!(network.edges[2].id, 2);
        assert_eq!(demands.demands[0].origin, 0);
        assert_eq!(demands.demands[0].destination, 1);
    }

    #[test]
    fn rejects_duplicate_edge_ids() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("edge_from", DataType::Utf8, false),
            Field::new("edge_to", DataType::Utf8, false),
            Field::new("edge_id", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "B"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["B", "C"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["AB", "AB"])) as ArrayRef,
            ],
        )
        .unwrap();

        let error = match prepare_network_batches(&[batch]) {
            Ok(_) => panic!("duplicate edge_id was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("duplicate edge_id"));
    }

    #[test]
    fn decodes_string_edge_paths_for_disruption() {
        let mut network = prepare_network_batches(&[string_network_batch()]).unwrap();
        let mut path_builder = ListBuilder::new(StringBuilder::new());
        path_builder.values().append_value("AC");
        path_builder.values().append_value("CB");
        path_builder.append(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("origin_id", DataType::Utf8, false),
            Field::new("destination_id", DataType::Utf8, false),
            Field::new("flow", DataType::Float64, false),
            Field::new(
                "edge_path",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("cost", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["B"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![7.0])) as ArrayRef,
                Arc::new(path_builder.finish()) as ArrayRef,
                Arc::new(Float64Array::from(vec![8.0])) as ArrayRef,
            ],
        )
        .unwrap();

        let inputs = prepare_disruption_inputs_batches(&[batch], &[1], &mut network).unwrap();

        assert_eq!(inputs.current_edge_flows, vec![0.0, 7.0, 7.0]);
        assert_eq!(inputs.affected_flows[0].edge_path, vec![1, 2]);
        assert_eq!(inputs.initial_costs_by_od, vec![(0, 1, 8.0)]);
    }

    #[test]
    fn writes_decoded_string_allocation_outputs() {
        let network = prepare_network_batches(&[string_network_batch()]).unwrap();
        let output = AllocationOutput {
            od_flows: vec![OdFlow {
                origin: 0,
                destination: 1,
                flow: 7.0,
                edge_path: vec![1, 2],
                cost: 8.0,
            }],
            network_flows: vec![EdgeFlow {
                edge_id: 1,
                edge_from: 0,
                edge_to: 2,
                cost: 3.0,
                capacity: Some(100.0),
                flow: 7.0,
            }],
            unassigned_od: Vec::new(),
        };

        let (od_flows, network_flows, _) = allocation_to_batches(&output, &network).unwrap();
        let origins = od_flows
            .column_by_name("origin_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let edge_ids = network_flows
            .column_by_name("edge_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(origins.value(0), "A");
        assert_eq!(edge_ids.value(0), "AC");
    }
}
