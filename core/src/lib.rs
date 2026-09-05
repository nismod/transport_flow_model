pub mod arrow_ffi;
pub mod core;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A network parsed once and reused across calls.
///
/// Every free function below parses its link table, interns ids and rebuilds
/// the graph from scratch. That is most of the cost of a call, and callers
/// that loop over origins or scenarios pay it every time round. This holds
/// the parsed form so the loop pays once.
#[pyclass(name = "PreparedNetwork", module = "transport_flow_model._core")]
struct PyPreparedNetwork {
    inner: arrow_ffi::PreparedNetwork,
}

#[pymethods]
impl PyPreparedNetwork {
    #[new]
    fn new(network: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = arrow_ffi::prepare_network_ffi(network).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_links(&self) -> usize {
        self.inner.n_links()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    fn allocate<'py>(
        &mut self,
        py: Python<'py>,
        od: &Bound<'py, PyAny>,
        capacity_constrained: bool,
        directed: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (od_flows, network_flows, unassigned_od) = self
            .inner
            .scoped(|network| {
                let demands = arrow_ffi::prepare_demands_ffi(od, network)?;
                let output = core::allocate(
                    &network.edges,
                    &demands.demands,
                    capacity_constrained,
                    directed,
                );
                arrow_ffi::allocation_to_ffi(py, &output, network)
            })
            .map_err(PyValueError::new_err)?;

        let result = PyDict::new(py);
        result.set_item("od_flows", od_flows)?;
        result.set_item("network_flows", network_flows)?;
        result.set_item("unassigned_od", unassigned_od)?;
        Ok(result)
    }

    fn skim<'py>(
        &mut self,
        py: Python<'py>,
        od_pairs: &Bound<'py, PyAny>,
        directed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .scoped(|network| {
                let pairs = arrow_ffi::prepare_skim_pairs_ffi(od_pairs, network)?;
                let costs = core::skim(&network.edges, &pairs, directed);
                arrow_ffi::skim_to_ffi(py, &pairs, &costs, network)
            })
            .map_err(PyValueError::new_err)
    }
}

/// A network and a baseline path set, both parsed once, for scenario loops.
///
/// Reading the path table dominates a scenario's cost and none of it depends
/// on which links fail, so a run over many scenarios should parse it once.
#[pyclass(name = "PreparedDisruption", module = "transport_flow_model._core")]
struct PyPreparedDisruption {
    network: arrow_ffi::PreparedNetwork,
    paths: arrow_ffi::PreparedOdPaths,
}

#[pymethods]
impl PyPreparedDisruption {
    #[new]
    fn new(network: &Bound<'_, PyAny>, od_flows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut network = arrow_ffi::prepare_network_ffi(network).map_err(PyValueError::new_err)?;
        let paths = arrow_ffi::prepare_od_paths_ffi(od_flows, &mut network)
            .map_err(PyValueError::new_err)?;
        Ok(Self { network, paths })
    }

    /// Reroute the baseline flows that use any of `failed_edges`.
    fn scenario<'py>(
        &self,
        py: Python<'py>,
        failed_edges: &Bound<'py, PyAny>,
        capacity_constrained: bool,
        directed: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let failed_edges = arrow_ffi::decode_failed_edges_ffi(failed_edges, &self.network)
            .map_err(PyValueError::new_err)?;
        let inputs = self.paths.for_failed_edges(&failed_edges);
        let output = core::disrupt_with_preprocessed(
            &self.network.edges,
            &inputs.affected_flows,
            self.paths.current_edge_flows(),
            &inputs.initial_costs_by_od,
            &failed_edges,
            capacity_constrained,
            directed,
        );
        let (rerouted_flows, network_flows, isolated_od, losses) =
            arrow_ffi::disruption_to_ffi(py, &output, &self.network)
                .map_err(PyValueError::new_err)?;

        let result = PyDict::new(py);
        result.set_item("rerouted_flows", rerouted_flows)?;
        result.set_item("network_flows", network_flows)?;
        result.set_item("isolated_od", isolated_od)?;
        result.set_item("losses", losses)?;
        Ok(result)
    }
}

#[pyfunction]
fn allocate_ffi<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyAny>,
    od: &Bound<'py, PyAny>,
    capacity_constrained: bool,
    directed: bool,
) -> PyResult<Bound<'py, PyDict>> {
    PyPreparedNetwork::new(network)?.allocate(py, od, capacity_constrained, directed)
}

#[pyfunction]
fn skim_ffi<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyAny>,
    od_pairs: &Bound<'py, PyAny>,
    directed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    PyPreparedNetwork::new(network)?.skim(py, od_pairs, directed)
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
    m.add_function(wrap_pyfunction!(shortest_paths_from_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(skim_ffi, m)?)?;
    m.add_class::<PyPreparedNetwork>()?;
    m.add_class::<PyPreparedDisruption>()?;
    Ok(())
}
