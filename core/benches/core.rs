use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use _core::core::{allocate, disrupt, shortest_path, Demand, Edge, OdFlow};

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

fn synthetic_network(size: usize) -> Vec<Edge> {
    let mut edges = Vec::new();
    for index in 0..size {
        edges.push(edge(index, index + 1, edges.len(), 1.0, Some(1_000.0)));
        if index + 2 <= size {
            edges.push(edge(index, index + 2, edges.len(), 3.0, Some(1_000.0)));
        }
    }
    edges
}

fn repeated_demands(count: usize, destination: usize) -> Vec<Demand> {
    (0..count)
        .map(|index| Demand {
            origin: index % 10,
            destination,
            flow: 10.0,
        })
        .collect()
}

fn bench_shortest_path(c: &mut Criterion) {
    let edges = synthetic_network(250);
    c.bench_function("shortest_path_250_edges", |b| {
        b.iter(|| shortest_path(black_box(&edges), black_box(0), black_box(250), true, None))
    });
}

fn bench_allocation(c: &mut Criterion) {
    let edges = synthetic_network(250);
    let demands = repeated_demands(100, 250);
    c.bench_function("allocate_unconstrained_250_edges_100_od", |b| {
        b.iter(|| allocate(black_box(&edges), black_box(&demands), false, true))
    });
}

fn bench_capacity_allocation(c: &mut Criterion) {
    let edges = vec![
        edge(0, 2, 0, 1.0, Some(100.0)),
        edge(1, 2, 1, 1.0, Some(100.0)),
        edge(2, 3, 2, 1.0, Some(500.0)),
    ];
    let demands: Vec<Demand> = (0..100)
        .map(|index| Demand {
            origin: if index % 2 == 0 { 0 } else { 1 },
            destination: 3,
            flow: 10.0,
        })
        .collect();
    c.bench_function("allocate_capacity_bottleneck_100_od", |b| {
        b.iter(|| allocate(black_box(&edges), black_box(&demands), true, true))
    });
}

fn bench_disruption(c: &mut Criterion) {
    let edges = vec![
        edge(0, 1, 0, 1.0, Some(1_000.0)),
        edge(1, 2, 1, 1.0, Some(1_000.0)),
        edge(0, 2, 2, 5.0, Some(1_000.0)),
    ];
    let existing: Vec<OdFlow> = (0..100)
        .map(|_| OdFlow {
            origin: 0,
            destination: 2,
            flow: 10.0,
            edge_path: vec![0, 1],
            cost: 2.0,
        })
        .collect();
    let failed_edges = vec![0];
    c.bench_function("disrupt_reroute_100_od", |b| {
        b.iter(|| {
            disrupt(
                black_box(&edges),
                black_box(&existing),
                black_box(&failed_edges),
                true,
                true,
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_shortest_path, bench_allocation, bench_capacity_allocation, bench_disruption
}
criterion_main!(benches);
