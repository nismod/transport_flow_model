use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub const CAPACITY_EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub id: String,
    pub cost: f64,
    pub capacity: Option<f64>,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Demand {
    pub origin: String,
    pub destination: String,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OdFlow {
    pub origin: String,
    pub destination: String,
    pub flow: f64,
    pub edge_path: Vec<String>,
    pub cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeFlow {
    pub edge_id: String,
    pub edge_from: String,
    pub edge_to: String,
    pub cost: f64,
    pub capacity: Option<f64>,
    pub flow: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loss {
    pub origin: String,
    pub destination: String,
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
    node: String,
    path: Vec<String>,
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

pub fn shortest_path(
    edges: &[Edge],
    origin: &str,
    destination: &str,
    directed: bool,
    residual_capacity: Option<&HashMap<String, f64>>,
) -> Option<(Vec<String>, f64)> {
    if origin == destination {
        return Some((Vec::new(), 0.0));
    }

    let mut adjacency: HashMap<&str, Vec<(&str, &str, f64, usize)>> = HashMap::new();
    let mut sequence = 0usize;
    for edge in edges {
        if residual_capacity
            .and_then(|capacity| capacity.get(&edge.id))
            .is_some_and(|available| *available <= CAPACITY_EPSILON)
        {
            continue;
        }
        adjacency.entry(edge.from.as_str()).or_default().push((
            edge.to.as_str(),
            edge.id.as_str(),
            edge.cost,
            sequence,
        ));
        sequence += 1;
        if !directed {
            adjacency.entry(edge.to.as_str()).or_default().push((
                edge.from.as_str(),
                edge.id.as_str(),
                edge.cost,
                sequence,
            ));
            sequence += 1;
        }
    }

    let mut heap = BinaryHeap::from([QueueState {
        cost: 0.0,
        order: 0,
        node: origin.to_string(),
        path: Vec::new(),
    }]);
    let mut best_cost = HashMap::from([(origin.to_string(), 0.0)]);
    let mut counter = 1usize;

    while let Some(state) = heap.pop() {
        if state.node == destination {
            return Some((state.path, state.cost));
        }
        if state.cost
            > best_cost.get(&state.node).copied().unwrap_or(f64::INFINITY) + CAPACITY_EPSILON
        {
            continue;
        }

        for (next_node, edge_id, edge_cost, order) in
            adjacency.get(state.node.as_str()).into_iter().flatten()
        {
            let next_cost = state.cost + edge_cost;
            if next_cost + CAPACITY_EPSILON
                < best_cost.get(*next_node).copied().unwrap_or(f64::INFINITY)
            {
                best_cost.insert((*next_node).to_string(), next_cost);
                let mut next_path = state.path.clone();
                next_path.push((*edge_id).to_string());
                heap.push(QueueState {
                    cost: next_cost,
                    order: order + counter,
                    node: (*next_node).to_string(),
                    path: next_path,
                });
                counter += 1;
            }
        }
    }

    None
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
    failed_edges: &[String],
    capacity_constrained: bool,
    directed: bool,
) -> DisruptionOutput {
    let failed_edge_set: HashSet<&str> = failed_edges.iter().map(String::as_str).collect();
    let affected_flows: Vec<OdFlow> = existing_flows
        .iter()
        .filter(|flow| {
            flow.edge_path
                .iter()
                .any(|edge_id| failed_edge_set.contains(edge_id.as_str()))
        })
        .cloned()
        .collect();

    let post_disruption_edges =
        edges_with_flows_removed_from_affected_paths(edges, existing_flows, &affected_flows);
    let mut post_disruption_edges: Vec<Edge> = post_disruption_edges
        .into_iter()
        .map(|mut edge| {
            if failed_edge_set.contains(edge.id.as_str()) {
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
        .filter(|edge| !failed_edge_set.contains(edge.id.as_str()))
        .cloned()
        .collect();
    let affected_demands: Vec<Demand> = affected_flows
        .iter()
        .map(|flow| Demand {
            origin: flow.origin.clone(),
            destination: flow.destination.clone(),
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

    // Keep the local variable mutable above so later congestion/speed-flow
    // updates can adjust post-disruption edge state in place.
    post_disruption_edges.clear();

    DisruptionOutput {
        rerouted_flows: allocation.od_flows,
        network_flows,
        isolated_od: allocation.unassigned_od,
        losses,
    }
}

fn allocate_unconstrained(edges: &[Edge], demands: &[Demand], directed: bool) -> AllocationOutput {
    let mut od_flows = Vec::new();
    let mut unassigned_od = Vec::new();

    for demand in demands {
        match shortest_path(edges, &demand.origin, &demand.destination, directed, None) {
            Some((edge_path, cost)) => od_flows.push(OdFlow {
                origin: demand.origin.clone(),
                destination: demand.destination.clone(),
                flow: demand.flow,
                edge_path,
                cost,
            }),
            None => unassigned_od.push(demand.clone()),
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
    let mut residual_capacity = initial_residual_capacity(edges);
    let mut pending = demands.to_vec();
    let mut allocated_rows = Vec::new();
    let mut unassigned_rows = Vec::new();

    while !pending.is_empty() {
        let mut route_rows = Vec::new();
        let mut next_pending = Vec::new();

        for demand in &pending {
            match shortest_path(
                edges,
                &demand.origin,
                &demand.destination,
                directed,
                Some(&residual_capacity),
            ) {
                Some((edge_path, cost)) => route_rows.push(OdFlow {
                    origin: demand.origin.clone(),
                    destination: demand.destination.clone(),
                    flow: demand.flow,
                    edge_path,
                    cost,
                }),
                None => unassigned_rows.push(demand.clone()),
            }
        }

        if route_rows.is_empty() {
            break;
        }

        let mut requested_by_edge: HashMap<String, f64> = HashMap::new();
        for route in &route_rows {
            for edge_id in &route.edge_path {
                *requested_by_edge.entry(edge_id.clone()).or_insert(0.0) += route.flow;
            }
        }

        let mut assigned_this_round = 0.0;
        let mut round_allocations = Vec::new();
        for route in route_rows {
            let requested_flow = route.flow;
            let mut assigned_flow = requested_flow;
            for edge_id in &route.edge_path {
                let requested_on_edge = requested_by_edge.get(edge_id).copied().unwrap_or(0.0);
                let available = residual_capacity.get(edge_id).copied().unwrap_or(0.0);
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
            for edge_id in &route.edge_path {
                if let Some(available) = residual_capacity.get_mut(edge_id) {
                    *available -= route.flow;
                    if *available < CAPACITY_EPSILON {
                        *available = 0.0;
                    }
                }
            }
        }

        if assigned_this_round <= CAPACITY_EPSILON {
            unassigned_rows.extend(next_pending);
            break;
        }

        pending = next_pending;
    }

    AllocationOutput {
        network_flows: network_flows_from_edges_and_od_flows(edges, &allocated_rows),
        od_flows: allocated_rows,
        unassigned_od: aggregate_demands(&unassigned_rows),
    }
}

fn initial_residual_capacity(edges: &[Edge]) -> HashMap<String, f64> {
    edges
        .iter()
        .map(|edge| {
            (
                edge.id.clone(),
                (edge.capacity.unwrap_or(f64::INFINITY) - edge.flow).max(0.0),
            )
        })
        .collect()
}

fn network_flows_from_edges_and_od_flows(edges: &[Edge], od_flows: &[OdFlow]) -> Vec<EdgeFlow> {
    let flow_by_edge = flow_by_edge(od_flows);
    edges
        .iter()
        .map(|edge| EdgeFlow {
            edge_id: edge.id.clone(),
            edge_from: edge.from.clone(),
            edge_to: edge.to.clone(),
            cost: edge.cost,
            capacity: edge.capacity,
            flow: edge.flow + flow_by_edge.get(&edge.id).copied().unwrap_or(0.0),
        })
        .collect()
}

fn edge_flows_from_edges(edges: &[Edge]) -> Vec<EdgeFlow> {
    edges
        .iter()
        .map(|edge| EdgeFlow {
            edge_id: edge.id.clone(),
            edge_from: edge.from.clone(),
            edge_to: edge.to.clone(),
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
    let current_flows = flow_by_edge(existing_flows);
    let affected_by_edge = flow_by_edge(affected_flows);
    let has_existing_edge_loads = edges.iter().any(|edge| edge.flow.abs() > CAPACITY_EPSILON);

    edges
        .iter()
        .map(|edge| {
            let base_flow = if has_existing_edge_loads {
                edge.flow
            } else {
                current_flows.get(&edge.id).copied().unwrap_or(0.0)
            };
            let mut next = edge.clone();
            next.flow =
                (base_flow - affected_by_edge.get(&edge.id).copied().unwrap_or(0.0)).max(0.0);
            next
        })
        .collect()
}

fn flow_by_edge(od_flows: &[OdFlow]) -> HashMap<String, f64> {
    let mut edge_flows = HashMap::new();
    for od_flow in od_flows {
        for edge_id in &od_flow.edge_path {
            *edge_flows.entry(edge_id.clone()).or_insert(0.0) += od_flow.flow;
        }
    }
    edge_flows
}

fn aggregate_demands(demands: &[Demand]) -> Vec<Demand> {
    let mut index: HashMap<(String, String), f64> = HashMap::new();
    for demand in demands {
        *index
            .entry((demand.origin.clone(), demand.destination.clone()))
            .or_insert(0.0) += demand.flow;
    }
    index
        .into_iter()
        .map(|((origin, destination), flow)| Demand {
            origin,
            destination,
            flow,
        })
        .collect()
}

fn losses_from_flows(initial: &[OdFlow], disrupted: &[OdFlow]) -> Vec<Loss> {
    let mut initial_costs: HashMap<(String, String), f64> = HashMap::new();
    for flow in initial {
        *initial_costs
            .entry((flow.origin.clone(), flow.destination.clone()))
            .or_insert(0.0) += flow.cost;
    }

    let mut disrupted_index: HashMap<(String, String), (f64, f64)> = HashMap::new();
    for flow in disrupted {
        let entry = disrupted_index
            .entry((flow.origin.clone(), flow.destination.clone()))
            .or_insert((0.0, 0.0));
        entry.0 += flow.flow;
        entry.1 += flow.cost;
    }

    disrupted_index
        .into_iter()
        .map(|((origin, destination), (flow, disrupted_cost))| {
            let initial_cost = initial_costs
                .get(&(origin.clone(), destination.clone()))
                .copied()
                .unwrap_or(0.0);
            Loss {
                origin,
                destination,
                flow,
                initial_cost,
                disrupted_cost,
                rerouting_loss: disrupted_cost - initial_cost,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(from: &str, to: &str, id: &str, cost: f64, capacity: Option<f64>) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
            id: id.to_string(),
            cost,
            capacity,
            flow: 0.0,
        }
    }

    #[test]
    fn finds_least_cost_path() {
        let edges = vec![
            edge("A", "B", "AB", 10.0, None),
            edge("A", "C", "AC", 3.0, None),
            edge("C", "B", "CB", 5.0, None),
        ];

        let (path, cost) = shortest_path(&edges, "A", "B", true, None).unwrap();

        assert_eq!(path, vec!["AC", "CB"]);
        assert_eq!(cost, 8.0);
    }

    #[test]
    fn respects_directed_flag() {
        let edges = vec![edge("A", "B", "AB", 1.0, None)];

        assert!(shortest_path(&edges, "B", "A", true, None).is_none());
        assert_eq!(
            shortest_path(&edges, "B", "A", false, None).unwrap().0,
            vec!["AB"]
        );
    }

    #[test]
    fn unconstrained_allocation_aggregates_network_flows() {
        let edges = vec![
            edge("A", "B", "AB", 1.0, None),
            edge("B", "C", "BC", 2.0, None),
            edge("A", "C", "AC", 5.0, None),
            edge("C", "D", "CD", 1.0, None),
            edge("B", "D", "BD", 10.0, None),
        ];
        let demands = vec![
            Demand {
                origin: "A".to_string(),
                destination: "C".to_string(),
                flow: 10.0,
            },
            Demand {
                origin: "B".to_string(),
                destination: "D".to_string(),
                flow: 6.0,
            },
        ];

        let result = allocate(&edges, &demands, false, true);
        let flows: HashMap<_, _> = result
            .network_flows
            .iter()
            .map(|flow| (flow.edge_id.as_str(), flow.flow))
            .collect();

        assert_eq!(flows["AB"], 10.0);
        assert_eq!(flows["BC"], 16.0);
        assert_eq!(flows["CD"], 6.0);
    }

    #[test]
    fn capacity_constrained_allocation_shares_bottleneck() {
        let edges = vec![
            edge("A", "C", "AC", 1.0, Some(100.0)),
            edge("B", "C", "BC", 1.0, Some(100.0)),
            edge("C", "D", "CD", 1.0, Some(10.0)),
        ];
        let demands = vec![
            Demand {
                origin: "A".to_string(),
                destination: "D".to_string(),
                flow: 10.0,
            },
            Demand {
                origin: "B".to_string(),
                destination: "D".to_string(),
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
            edge("A", "B", "AB", 1.0, Some(100.0)),
            edge("B", "C", "BC", 1.0, Some(100.0)),
            edge("A", "C", "AC", 5.0, Some(100.0)),
        ];
        let existing = vec![OdFlow {
            origin: "A".to_string(),
            destination: "C".to_string(),
            flow: 10.0,
            edge_path: vec!["AB".to_string(), "BC".to_string()],
            cost: 2.0,
        }];

        let result = disrupt(&edges, &existing, &["AB".to_string()], true, true);

        assert_eq!(result.rerouted_flows[0].edge_path, vec!["AC"]);
        assert_eq!(result.rerouted_flows[0].cost, 5.0);
        assert!(result.isolated_od.is_empty());
        assert_eq!(result.losses[0].rerouting_loss, 3.0);
    }

    #[test]
    fn disruption_isolates_when_no_path_remains() {
        let edges = vec![
            edge("A", "B", "AB", 1.0, Some(100.0)),
            edge("B", "C", "BC", 1.0, Some(100.0)),
        ];
        let existing = vec![OdFlow {
            origin: "A".to_string(),
            destination: "C".to_string(),
            flow: 10.0,
            edge_path: vec!["AB".to_string(), "BC".to_string()],
            cost: 2.0,
        }];

        let result = disrupt(&edges, &existing, &["AB".to_string()], true, true);

        assert!(result.rerouted_flows.is_empty());
        assert_eq!(result.isolated_od[0].flow, 10.0);
    }
}
