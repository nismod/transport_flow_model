pub mod arrow_ffi;
pub mod core;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn allocate_ffi<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyAny>,
    od: &Bound<'py, PyAny>,
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut network = arrow_ffi::prepare_network_ffi(network).map_err(PyValueError::new_err)?;
    let demands =
        arrow_ffi::prepare_demands_ffi(od, &mut network).map_err(PyValueError::new_err)?;
    let output = core::allocate(
        &network.edges,
        &demands.demands,
        capacity_constrained,
        directed,
    );
    let (od_flows, network_flows, unassigned_od) =
        arrow_ffi::allocation_to_ffi(py, &output, &network).map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("od_flows", od_flows)?;
    result.set_item("network_flows", network_flows)?;
    result.set_item("unassigned_od", unassigned_od)?;
    Ok(result)
}

#[pyfunction]
fn disrupt_ffi<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyAny>,
    od_flows: &Bound<'py, PyAny>,
    failed_edges: &Bound<'py, PyAny>,
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut network = arrow_ffi::prepare_network_ffi(network).map_err(PyValueError::new_err)?;
    let od_flows = arrow_ffi::prepare_od_flows_ffi(od_flows, failed_edges, &mut network)
        .map_err(PyValueError::new_err)?;
    let output = core::disrupt_with_preprocessed(
        &network.edges,
        &od_flows.inputs.affected_flows,
        &od_flows.inputs.current_edge_flows,
        &od_flows.inputs.initial_costs_by_od,
        &od_flows.failed_edges,
        capacity_constrained,
        directed,
    );
    let (rerouted_flows, network_flows, isolated_od, losses) =
        arrow_ffi::disruption_to_ffi(py, &output, &network).map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("rerouted_flows", rerouted_flows)?;
    result.set_item("network_flows", network_flows)?;
    result.set_item("isolated_od", isolated_od)?;
    result.set_item("losses", losses)?;
    Ok(result)
}

#[pyfunction]
fn shortest_paths_from_ffi<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyAny>,
    origin: usize,
    directed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let edges = arrow_ffi::read_network_ffi(network).map_err(PyValueError::new_err)?;
    let paths = core::shortest_paths_from(&edges, origin, directed);
    arrow_ffi::shortest_paths_to_ffi(py, &paths).map_err(PyValueError::new_err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(disrupt_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_paths_from_ffi, m)?)?;
    Ok(())
}
