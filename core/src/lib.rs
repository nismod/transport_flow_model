pub mod arrow_ffi;
pub mod arrow_ipc;
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
    let network = arrow_ffi::read_network_ffi(network).map_err(PyValueError::new_err)?;
    let demands = arrow_ffi::read_demands_ffi(od).map_err(PyValueError::new_err)?;
    let output = core::allocate(&network, &demands, capacity_constrained, directed);
    let (od_flows, network_flows, unassigned_od) =
        arrow_ffi::allocation_to_ffi(py, &output).map_err(PyValueError::new_err)?;

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
    failed_edges: Vec<usize>,
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let network = arrow_ffi::read_network_ffi(network).map_err(PyValueError::new_err)?;
    let network_edge_capacity = network
        .iter()
        .map(|edge| edge.id)
        .max()
        .map_or(0, |edge_id| edge_id + 1);
    let disruption_inputs =
        arrow_ffi::read_disruption_inputs_ffi(od_flows, &failed_edges, network_edge_capacity)
            .map_err(PyValueError::new_err)?;
    let output = core::disrupt_with_preprocessed(
        &network,
        &disruption_inputs.affected_flows,
        &disruption_inputs.current_edge_flows,
        &failed_edges,
        capacity_constrained,
        directed,
    );
    let (rerouted_flows, network_flows, isolated_od, losses) =
        arrow_ffi::disruption_to_ffi(py, &output).map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("rerouted_flows", rerouted_flows)?;
    result.set_item("network_flows", network_flows)?;
    result.set_item("isolated_od", isolated_od)?;
    result.set_item("losses", losses)?;
    Ok(result)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(disrupt_ffi, m)?)?;
    Ok(())
}
