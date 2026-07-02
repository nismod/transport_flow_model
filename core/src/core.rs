use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub const CAPACITY_EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub id: usize,
    pub cost: f64,
    pub capacity: Option<f64>,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Demand {
    pub origin: usize,
    pub destination: usize,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OdFlow {
    pub origin: usize,
    pub destination: usize,
    pub flow: f64,
    pub edge_path: Vec<usize>,
    pub cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeFlow {
    pub edge_id: usize,
    pub edge_from: usize,
    pub edge_to: usize,
    pub cost: f64,
    pub capacity: Option<f64>,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loss {
    pub origin: usize,
    pub destination: usize,
    pub flow: f64,
    pub initial_cost: f64,
    pub disrupted_cost: f64,
    pub rerouting_loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AllocationOutput {
    pub od_flows: Vec<OdFlow>,
    pub network_flows: Vec<EdgeFlow>,
    pub unassigned_od: Vec<Demand>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DisruptionOutput {
    pub rerouted_flows: Vec<OdFlow>,
    pub network_flows: Vec<EdgeFlow>,
    pub isolated_od: Vec<Demand>,
    pub losses: Vec<Loss>,
}

#[derive(Clone, Debug)]
struct QueueState {
    cost: f64,
    order: usize,
    node: usize,
    path: Vec<usize>,
}

impl Eq for QueueState {}

impl PartialEq for QueueState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.order == other.order
    }
}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.order.cmp(&self.order))
    }
}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
struct AdjacentEdge {
    next_node: usize,
    edge_index: usize,
    edge_cost: f64,
    order: usize,
}

#[derive(Debug)]
struct Graph<'a> {
    edges: &'a [Edge],
    adjacency: Vec<Vec<AdjacentEdge>>,
}

impl<'a> Graph<'a> {
    fn new(edges: &'a [Edge], directed: bool) -> Self {
        let node_count = max_node_id(edges).map_or(0, |node_id| node_id + 1);
        let mut adjacency = vec![Vec::new(); node_count];
        let mut sequence = 0usize;

        for (edge_index, edge) in edges.iter().enumerate() {
            adjacency[edge.from].push(AdjacentEdge {
                next_node: edge.to,
                edge_index,
                edge_cost: edge.cost,
                order: sequence,
            });
            sequence += 1;

            if !directed {
                adjacency[edge.to].push(AdjacentEdge {
                    next_node: edge.from,
                    edge_index,
                    edge_cost: edge.cost,
                    order: sequence,
                });
                sequence += 1;
            }
        }

        Self { edges, adjacency }
    }

    fn shortest_path(
        &self,
        origin: usize,
        destination: usize,
        residual_capacity: Option<&[f64]>,
    ) -> Option<(Vec<usize>, f64)> {
        if origin == destination {
            return Some((Vec::new(), 0.0));
        }
        if origin >= self.adjacency.len() || destination >= self.adjacency.len() {
            return None;
        }

        let mut heap = BinaryHeap::from([QueueState {
            cost: 0.0,
            order: 0,
            node: origin,
            path: Vec::new(),
        }]);
        let mut best_cost = vec![f64::INFINITY; self.adjacency.len()];
        best_cost[origin] = 0.0;
        let mut counter = 1usize;

        while let Some(state) = heap.pop() {
            if state.node == destination {
                return Some((state.path, state.cost));
            }
            if state.cost > best_cost[state.node] + CAPACITY_EPSILON {
                continue;
            }

            for adjacent in &self.adjacency[state.node] {
                if residual_capacity
                    .is_some_and(|capacity| capacity[adjacent.edge_index] <= CAPACITY_EPSILON)
                {
                    continue;
                }

                let next_cost = state.cost + adjacent.edge_cost;
                if next_cost + CAPACITY_EPSILON < best_cost[adjacent.next_node] {
                    best_cost[adjacent.next_node] = next_cost;
                    let mut next_path = state.path.clone();
                    next_path.push(adjacent.edge_index);
                    heap.push(QueueState {
                        cost: next_cost,
                        order: adjacent.order + counter,
                        node: adjacent.next_node,
                        path: next_path,
                    });
                    counter += 1;
                }
            }
        }

        None
    }

    fn single_source_shortest_paths(
        &self,
        origin: usize,
        residual_capacity: Option<&[f64]>,
    ) -> HashMap<usize, (Vec<usize>, f64)> {
        if origin >= self.adjacency.len() {
            return HashMap::new();
        }

        let mut paths = HashMap::new();
        paths.insert(origin, (Vec::new(), 0.0));

        let mut heap = BinaryHeap::from([QueueState {
            cost: 0.0,
            order: 0,
            node: origin,
            path: Vec::new(),
        }]);
        let mut best_cost = vec![f64::INFINITY; self.adjacency.len()];
        best_cost[origin] = 0.0;
        let mut counter = 1usize;

        while let Some(state) = heap.pop() {
            if state.cost > best_cost[state.node] + CAPACITY_EPSILON {
                continue;
            }

            for adjacent in &self.adjacency[state.node] {
                if residual_capacity
                    .is_some_and(|capacity| capacity[adjacent.edge_index] <= CAPACITY_EPSILON)
                {
                    continue;
                }

                let next_cost = state.cost + adjacent.edge_cost;
                if next_cost + CAPACITY_EPSILON < best_cost[adjacent.next_node] {
                    best_cost[adjacent.next_node] = next_cost;
                    let mut next_path = state.path.clone();
                    next_path.push(adjacent.edge_index);
                    paths.insert(adjacent.next_node, (next_path.clone(), next_cost));
                    heap.push(QueueState {
                        cost: next_cost,
                        order: adjacent.order + counter,
                        node: adjacent.next_node,
                        path: next_path,
                    });
                    counter += 1;
                }
            }
        }

        paths
    }

    fn edge_path_ids(&self, edge_path: &[usize]) -> Vec<usize> {
        edge_path
            .iter()
            .map(|edge_index| self.edges[*edge_index].id)
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Route {
    origin: usize,
    destination: usize,
    flow: f64,
    edge_path: Vec<usize>,
    cost: f64,
}

impl Route {
    fn into_od_flow(self, graph: &Graph<'_>) -> OdFlow {
        OdFlow {
            origin: self.origin,
            destination: self.destination,
            flow: self.flow,
            edge_path: graph.edge_path_ids(&self.edge_path),
            cost: self.cost,
        }
    }
}

pub fn shortest_path(
    edges: &[Edge],
    origin: usize,
    destination: usize,
    directed: bool,
    residual_capacity: Option<&[f64]>,
) -> Option<(Vec<usize>, f64)> {
    let graph = Graph::new(edges, directed);
    graph
        .shortest_path(origin, destination, residual_capacity)
        .map(|(path, cost)| (graph.edge_path_ids(&path), cost))
}

pub fn allocate(
    edges: &[Edge],
    demands: &[Demand],
    capacity_constrained: bool,
    directed: bool,
) -> AllocationOutput {
    if capacity_constrained {
        allocate_capacity_constrained(edges, demands, directed)
    } else {
        allocate_unconstrained(edges, demands, directed)
    }
}

pub fn disrupt(
    edges: &[Edge],
    existing_flows: &[OdFlow],
    failed_edges: &[usize],
    capacity_constrained: bool,
    directed: bool,
) -> DisruptionOutput {
    let failed_edge_flags = edge_flags(failed_edges, max_edge_id(edges, existing_flows));
    let affected_flows: Vec<OdFlow> = existing_flows
        .iter()
        .filter(|flow| {
            flow.edge_path
                .iter()
                .any(|edge_id| failed_edge_flags.get(*edge_id).copied().unwrap_or(false))
        })
        .cloned()
        .collect();

    let post_disruption_edges =
        edges_with_flows_removed_from_affected_paths(edges, existing_flows, &affected_flows);
    let mut post_disruption_edges: Vec<Edge> = post_disruption_edges
        .into_iter()
        .map(|mut edge| {
            if failed_edge_flags.get(edge.id).copied().unwrap_or(false) {
                edge.flow = 0.0;
            }
            edge
        })
        .collect();

    if affected_flows.is_empty() {
        return DisruptionOutput {
            rerouted_flows: Vec::new(),
            network_flows: edge_flows_from_edges(&post_disruption_edges),
            isolated_od: Vec::new(),
            losses: Vec::new(),
        };
    }

    let reroute_source_edges = if capacity_constrained {
        &post_disruption_edges
    } else {
        edges
    };
    let reroute_edges: Vec<Edge> = reroute_source_edges
        .iter()
        .filter(|edge| !failed_edge_flags.get(edge.id).copied().unwrap_or(false))
        .cloned()
        .collect();
    let affected_demands: Vec<Demand> = affected_flows
        .iter()
        .map(|flow| Demand {
            origin: flow.origin,
            destination: flow.destination,
            flow: flow.flow,
        })
        .collect();
    let allocation = allocate(
        &reroute_edges,
        &affected_demands,
        capacity_constrained,
        directed,
    );
    let network_flows =
        network_flows_from_edges_and_od_flows(&post_disruption_edges, &allocation.od_flows);
    let losses = losses_from_flows(&affected_flows, &allocation.od_flows);

    post_disruption_edges.clear();

    DisruptionOutput {
        rerouted_flows: allocation.od_flows,
        network_flows,
        isolated_od: allocation.unassigned_od,
        losses,
    }
}

fn allocate_unconstrained(edges: &[Edge], demands: &[Demand], directed: bool) -> AllocationOutput {
    let graph = Graph::new(edges, directed);
    let mut od_flows = Vec::new();
    let mut unassigned_od = Vec::new();

    // Group demands by origin
    let mut demands_by_origin: HashMap<usize, Vec<&Demand>> = HashMap::new();
    for demand in demands {
        demands_by_origin
            .entry(demand.origin)
            .or_insert_with(Vec::new)
            .push(demand);
    }

    // Process each origin with single-source shortest paths
    for (origin, origin_demands) in demands_by_origin {
        let paths = graph.single_source_shortest_paths(origin, None);

        for demand in origin_demands {
            match paths.get(&demand.destination) {
                Some((edge_path, cost)) => {
                    od_flows.push(OdFlow {
                        origin: demand.origin,
                        destination: demand.destination,
                        flow: demand.flow,
                        edge_path: graph.edge_path_ids(edge_path),
                        cost: *cost,
                    });
                }
                None => unassigned_od.push(demand.clone()),
            }
        }
    }

    AllocationOutput {
        network_flows: network_flows_from_edges_and_od_flows(edges, &od_flows),
        od_flows,
        unassigned_od,
    }
}

fn allocate_capacity_constrained(
    edges: &[Edge],
    demands: &[Demand],
    directed: bool,
) -> AllocationOutput {
    let graph = Graph::new(edges, directed);
    let mut residual_capacity = initial_residual_capacity(edges);
    let mut pending = demands.to_vec();
    let mut allocated_rows: Vec<Route> = Vec::new();
    let mut unassigned_rows = Vec::new();

    while !pending.is_empty() {
        // Group pending demands by origin
        let mut demands_by_origin: HashMap<usize, Vec<Demand>> = HashMap::new();
        for demand in &pending {
            demands_by_origin
                .entry(demand.origin)
                .or_insert_with(Vec::new)
                .push(demand.clone());
        }

        let mut route_rows = Vec::new();
        let mut next_pending = Vec::new();

        // Process each origin with single-source shortest paths
        for (origin, origin_demands) in demands_by_origin {
            let paths = graph.single_source_shortest_paths(origin, Some(&residual_capacity));

            for demand in origin_demands {
                match paths.get(&demand.destination) {
                    Some((edge_path, cost)) => {
                        route_rows.push(Route {
                            origin: demand.origin,
                            destination: demand.destination,
                            flow: demand.flow,
                            edge_path: edge_path.clone(),
                            cost: *cost,
                        });
                    }
                    None => unassigned_rows.push(demand),
                }
            }
        }

        if route_rows.is_empty() {
            break;
        }

        let mut requested_by_edge = vec![0.0; edges.len()];
        for route in &route_rows {
            for edge_index in &route.edge_path {
                requested_by_edge[*edge_index] += route.flow;
            }
        }

        let mut assigned_this_round = 0.0;
        let mut round_allocations = Vec::new();
        for route in route_rows {
            let requested_flow = route.flow;
            let mut assigned_flow = requested_flow;
            for edge_index in &route.edge_path {
                let requested_on_edge = requested_by_edge[*edge_index];
                let available = residual_capacity[*edge_index];
                if requested_on_edge > available + CAPACITY_EPSILON {
                    assigned_flow =
                        assigned_flow.min(requested_flow * available / requested_on_edge);
                }
            }

            if assigned_flow > CAPACITY_EPSILON {
                let mut assigned_route = route.clone();
                assigned_route.flow = assigned_flow;
                allocated_rows.push(assigned_route.clone());
                round_allocations.push(assigned_route);
                assigned_this_round += assigned_flow;
            }

            let residual_flow = requested_flow - assigned_flow;
            if residual_flow > CAPACITY_EPSILON {
                next_pending.push(Demand {
                    origin: route.origin,
                    destination: route.destination,
                    flow: residual_flow,
                });
            }
        }

        for route in &round_allocations {
            for edge_index in &route.edge_path {
                residual_capacity[*edge_index] -= route.flow;
                if residual_capacity[*edge_index] < CAPACITY_EPSILON {
                    residual_capacity[*edge_index] = 0.0;
                }
            }
        }

        if assigned_this_round <= CAPACITY_EPSILON {
            unassigned_rows.extend(next_pending);
            break;
        }

        pending = next_pending;
    }

    let od_flows = allocated_rows
        .into_iter()
        .map(|route| route.into_od_flow(&graph))
        .collect::<Vec<_>>();

    AllocationOutput {
        network_flows: network_flows_from_edges_and_od_flows(edges, &od_flows),
        od_flows,
        unassigned_od: aggregate_demands(&unassigned_rows),
    }
}

fn initial_residual_capacity(edges: &[Edge]) -> Vec<f64> {
    edges
        .iter()
        .map(|edge| (edge.capacity.unwrap_or(f64::INFINITY) - edge.flow).max(0.0))
        .collect()
}

fn network_flows_from_edges_and_od_flows(edges: &[Edge], od_flows: &[OdFlow]) -> Vec<EdgeFlow> {
    let edge_flow_capacity = max_edge_id(edges, od_flows).map_or(0, |edge_id| edge_id + 1);
    let flow_by_edge = flow_by_edge(od_flows, edge_flow_capacity);
    edges
        .iter()
        .map(|edge| EdgeFlow {
            edge_id: edge.id,
            edge_from: edge.from,
            edge_to: edge.to,
            cost: edge.cost,
            capacity: edge.capacity,
            flow: edge.flow + flow_by_edge.get(edge.id).copied().unwrap_or(0.0),
        })
        .collect()
}

fn edge_flows_from_edges(edges: &[Edge]) -> Vec<EdgeFlow> {
    edges
        .iter()
        .map(|edge| EdgeFlow {
            edge_id: edge.id,
            edge_from: edge.from,
            edge_to: edge.to,
            cost: edge.cost,
            capacity: edge.capacity,
            flow: edge.flow,
        })
        .collect()
}

fn edges_with_flows_removed_from_affected_paths(
    edges: &[Edge],
    existing_flows: &[OdFlow],
    affected_flows: &[OdFlow],
) -> Vec<Edge> {
    let edge_flow_capacity = max_edge_id(edges, existing_flows)
        .into_iter()
        .chain(max_edge_id(edges, affected_flows))
        .max()
        .map_or(0, |edge_id| edge_id + 1);
    let current_flows = flow_by_edge(existing_flows, edge_flow_capacity);
    let affected_by_edge = flow_by_edge(affected_flows, edge_flow_capacity);
    let has_existing_edge_loads = edges.iter().any(|edge| edge.flow.abs() > CAPACITY_EPSILON);

    edges
        .iter()
        .map(|edge| {
            let base_flow = if has_existing_edge_loads {
                edge.flow
            } else {
                current_flows.get(edge.id).copied().unwrap_or(0.0)
            };
            let mut next = edge.clone();
            next.flow =
                (base_flow - affected_by_edge.get(edge.id).copied().unwrap_or(0.0)).max(0.0);
            next
        })
        .collect()
}

fn flow_by_edge(od_flows: &[OdFlow], edge_count: usize) -> Vec<f64> {
    let mut edge_flows = vec![0.0; edge_count];
    for od_flow in od_flows {
        for edge_id in &od_flow.edge_path {
            if let Some(flow) = edge_flows.get_mut(*edge_id) {
                *flow += od_flow.flow;
            }
        }
    }
    edge_flows
}

fn aggregate_demands(demands: &[Demand]) -> Vec<Demand> {
    if demands.is_empty() {
        return Vec::new();
    }

    let mut rows = demands.to_vec();
    rows.sort_by_key(|demand| (demand.origin, demand.destination));

    let mut aggregated: Vec<Demand> = Vec::new();
    for demand in rows {
        if let Some(last) = aggregated.last_mut() {
            if last.origin == demand.origin && last.destination == demand.destination {
                last.flow += demand.flow;
                continue;
            }
        }
        aggregated.push(demand);
    }
    aggregated
}

fn losses_from_flows(initial: &[OdFlow], disrupted: &[OdFlow]) -> Vec<Loss> {
    if disrupted.is_empty() {
        return Vec::new();
    }

    let mut initial_costs = initial
        .iter()
        .map(|flow| (flow.origin, flow.destination, flow.cost))
        .collect::<Vec<_>>();
    initial_costs.sort_by_key(|(origin, destination, _)| (*origin, *destination));
    let initial_costs = sum_costs(initial_costs);

    let mut disrupted_rows = disrupted
        .iter()
        .map(|flow| (flow.origin, flow.destination, flow.flow, flow.cost))
        .collect::<Vec<_>>();
    disrupted_rows.sort_by_key(|(origin, destination, _, _)| (*origin, *destination));

    let mut losses = Vec::new();
    for (origin, destination, flow, disrupted_cost) in sum_flows_and_costs(disrupted_rows) {
        let initial_cost = lookup_cost(&initial_costs, origin, destination);
        losses.push(Loss {
            origin,
            destination,
            flow,
            initial_cost,
            disrupted_cost,
            rerouting_loss: disrupted_cost - initial_cost,
        });
    }
    losses
}

fn sum_costs(rows: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
    let mut summed: Vec<(usize, usize, f64)> = Vec::new();
    for (origin, destination, cost) in rows {
        if let Some(last) = summed.last_mut() {
            if last.0 == origin && last.1 == destination {
                last.2 += cost;
                continue;
            }
        }
        summed.push((origin, destination, cost));
    }
    summed
}

fn sum_flows_and_costs(rows: Vec<(usize, usize, f64, f64)>) -> Vec<(usize, usize, f64, f64)> {
    let mut summed: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (origin, destination, flow, cost) in rows {
        if let Some(last) = summed.last_mut() {
            if last.0 == origin && last.1 == destination {
                last.2 += flow;
                last.3 += cost;
                continue;
            }
        }
        summed.push((origin, destination, flow, cost));
    }
    summed
}

fn lookup_cost(rows: &[(usize, usize, f64)], origin: usize, destination: usize) -> f64 {
    match rows.binary_search_by_key(
        &(origin, destination),
        |(row_origin, row_destination, _)| (*row_origin, *row_destination),
    ) {
        Ok(index) => rows[index].2,
        Err(_) => 0.0,
    }
}

fn edge_flags(edge_ids: &[usize], max_edge_id: Option<usize>) -> Vec<bool> {
    let edge_count = edge_ids
        .iter()
        .copied()
        .max()
        .into_iter()
        .chain(max_edge_id)
        .max()
        .map_or(0, |edge_id| edge_id + 1);
    let mut flags = vec![false; edge_count];
    for edge_id in edge_ids {
        if let Some(flag) = flags.get_mut(*edge_id) {
            *flag = true;
        }
    }
    flags
}

fn max_node_id(edges: &[Edge]) -> Option<usize> {
    edges.iter().flat_map(|edge| [edge.from, edge.to]).max()
}

fn max_edge_id(edges: &[Edge], od_flows: &[OdFlow]) -> Option<usize> {
    edges
        .iter()
        .map(|edge| edge.id)
        .chain(
            od_flows
                .iter()
                .flat_map(|od_flow| od_flow.edge_path.iter().copied()),
        )
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(from: usize, to: usize, id: usize, cost: f64, capacity: Option<f64>) -> Edge {
        Edge {
            from,
            to,
            id,
            cost,
            capacity,
            flow: 0.0,
        }
    }

    #[test]
    fn finds_least_cost_path() {
        let edges = vec![
            edge(0, 1, 0, 10.0, None),
            edge(0, 2, 1, 3.0, None),
            edge(2, 1, 2, 5.0, None),
        ];

        let (path, cost) = shortest_path(&edges, 0, 1, true, None).unwrap();

        assert_eq!(path, vec![1, 2]);
        assert_eq!(cost, 8.0);
    }

    #[test]
    fn respects_directed_flag() {
        let edges = vec![edge(0, 1, 0, 1.0, None)];

        assert!(shortest_path(&edges, 1, 0, true, None).is_none());
        assert_eq!(shortest_path(&edges, 1, 0, false, None).unwrap().0, vec![0]);
    }

    #[test]
    fn unconstrained_allocation_aggregates_network_flows() {
        let edges = vec![
            edge(0, 1, 0, 1.0, None),
            edge(1, 2, 1, 2.0, None),
            edge(0, 2, 2, 5.0, None),
            edge(2, 3, 3, 1.0, None),
            edge(1, 3, 4, 10.0, None),
        ];
        let demands = vec![
            Demand {
                origin: 0,
                destination: 2,
                flow: 10.0,
            },
            Demand {
                origin: 1,
                destination: 3,
                flow: 6.0,
            },
        ];

        let result = allocate(&edges, &demands, false, true);
        let flows = result
            .network_flows
            .iter()
            .map(|flow| flow.flow)
            .collect::<Vec<_>>();

        assert_eq!(flows[0], 10.0);
        assert_eq!(flows[1], 16.0);
        assert_eq!(flows[3], 6.0);
    }

    #[test]
    fn capacity_constrained_allocation_shares_bottleneck() {
        let edges = vec![
            edge(0, 2, 0, 1.0, Some(100.0)),
            edge(1, 2, 1, 1.0, Some(100.0)),
            edge(2, 3, 2, 1.0, Some(10.0)),
        ];
        let demands = vec![
            Demand {
                origin: 0,
                destination: 3,
                flow: 10.0,
            },
            Demand {
                origin: 1,
                destination: 3,
                flow: 10.0,
            },
        ];

        let result = allocate(&edges, &demands, true, true);

        assert_eq!(result.od_flows.len(), 2);
        assert_eq!(result.od_flows[0].flow, 5.0);
        assert_eq!(result.od_flows[1].flow, 5.0);
        assert_eq!(result.unassigned_od.len(), 2);
    }

    #[test]
    fn disruption_reroutes_to_available_path() {
        let edges = vec![
            edge(0, 1, 0, 1.0, Some(100.0)),
            edge(1, 2, 1, 1.0, Some(100.0)),
            edge(0, 2, 2, 5.0, Some(100.0)),
        ];
        let existing = vec![OdFlow {
            origin: 0,
            destination: 2,
            flow: 10.0,
            edge_path: vec![0, 1],
            cost: 2.0,
        }];

        let result = disrupt(&edges, &existing, &[0], true, true);

        assert_eq!(result.rerouted_flows[0].edge_path, vec![2]);
        assert_eq!(result.rerouted_flows[0].cost, 5.0);
        assert!(result.isolated_od.is_empty());
        assert_eq!(result.losses[0].rerouting_loss, 3.0);
    }

    #[test]
    fn disruption_isolates_when_no_path_remains() {
        let edges = vec![
            edge(0, 1, 0, 1.0, Some(100.0)),
            edge(1, 2, 1, 1.0, Some(100.0)),
        ];
        let existing = vec![OdFlow {
            origin: 0,
            destination: 2,
            flow: 10.0,
            edge_path: vec![0, 1],
            cost: 2.0,
        }];

        let result = disrupt(&edges, &existing, &[0], true, true);

        assert!(result.rerouted_flows.is_empty());
        assert_eq!(result.isolated_od[0].flow, 10.0);
    }
}
