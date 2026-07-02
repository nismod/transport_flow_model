pub mod arrow_ipc;
pub mod core;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn allocate_ipc<'py>(
    py: Python<'py>,
    network_ipc: &[u8],
    od_ipc: &[u8],
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let network = arrow_ipc::read_network_ipc(network_ipc).map_err(PyValueError::new_err)?;
    let demands = arrow_ipc::read_demands_ipc(od_ipc).map_err(PyValueError::new_err)?;
    let output = core::allocate(&network, &demands, capacity_constrained, directed);
    let (od_flows, network_flows, unassigned_od) =
        arrow_ipc::allocation_to_ipc(&output).map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("od_flows", PyBytes::new(py, &od_flows))?;
    result.set_item("network_flows", PyBytes::new(py, &network_flows))?;
    result.set_item("unassigned_od", PyBytes::new(py, &unassigned_od))?;
    Ok(result)
}

#[pyfunction]
fn disrupt_ipc<'py>(
    py: Python<'py>,
    network_ipc: &[u8],
    od_flows_ipc: &[u8],
    failed_edges: Vec<String>,
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let network = arrow_ipc::read_network_ipc(network_ipc).map_err(PyValueError::new_err)?;
    let od_flows = arrow_ipc::read_od_flows_ipc(od_flows_ipc).map_err(PyValueError::new_err)?;
    let output = core::disrupt(
        &network,
        &od_flows,
        &failed_edges,
        capacity_constrained,
        directed,
    );
    let (rerouted_flows, network_flows, isolated_od, losses) =
        arrow_ipc::disruption_to_ipc(&output).map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("rerouted_flows", PyBytes::new(py, &rerouted_flows))?;
    result.set_item("network_flows", PyBytes::new(py, &network_flows))?;
    result.set_item("isolated_od", PyBytes::new(py, &isolated_od))?;
    result.set_item("losses", PyBytes::new(py, &losses))?;
    Ok(result)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(disrupt_ipc, m)?)?;
    Ok(())
}
