use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use _rust::core::{allocate, disrupt, shortest_path, Demand, Edge, OdFlow};

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

fn synthetic_network(size: usize) -> Vec<Edge> {
    let mut edges = Vec::new();
    for index in 0..size {
        let from = format!("N{index}");
        let to = format!("N{}", index + 1);
        edges.push(edge(&from, &to, &format!("E{index}"), 1.0, Some(1_000.0)));
        if index + 2 <= size {
            let skip_to = format!("N{}", index + 2);
            edges.push(edge(
                &from,
                &skip_to,
                &format!("S{index}"),
                3.0,
                Some(1_000.0),
            ));
        }
    }
    edges
}

fn repeated_demands(count: usize, destination: &str) -> Vec<Demand> {
    (0..count)
        .map(|index| Demand {
            origin: format!("N{}", index % 10),
            destination: destination.to_string(),
            flow: 10.0,
        })
        .collect()
}

fn bench_shortest_path(c: &mut Criterion) {
    let edges = synthetic_network(250);
    c.bench_function("shortest_path_250_edges", |b| {
        b.iter(|| {
            shortest_path(
                black_box(&edges),
                black_box("N0"),
                black_box("N250"),
                true,
                None,
            )
        })
    });
}

fn bench_allocation(c: &mut Criterion) {
    let edges = synthetic_network(250);
    let demands = repeated_demands(100, "N250");
    c.bench_function("allocate_unconstrained_250_edges_100_od", |b| {
        b.iter(|| allocate(black_box(&edges), black_box(&demands), false, true))
    });
}

fn bench_capacity_allocation(c: &mut Criterion) {
    let edges = vec![
        edge("A", "C", "AC", 1.0, Some(100.0)),
        edge("B", "C", "BC", 1.0, Some(100.0)),
        edge("C", "D", "CD", 1.0, Some(500.0)),
    ];
    let demands: Vec<Demand> = (0..100)
        .map(|index| Demand {
            origin: if index % 2 == 0 { "A" } else { "B" }.to_string(),
            destination: "D".to_string(),
            flow: 10.0,
        })
        .collect();
    c.bench_function("allocate_capacity_bottleneck_100_od", |b| {
        b.iter(|| allocate(black_box(&edges), black_box(&demands), true, true))
    });
}

fn bench_disruption(c: &mut Criterion) {
    let edges = vec![
        edge("A", "B", "AB", 1.0, Some(1_000.0)),
        edge("B", "C", "BC", 1.0, Some(1_000.0)),
        edge("A", "C", "AC", 5.0, Some(1_000.0)),
    ];
    let existing: Vec<OdFlow> = (0..100)
        .map(|_| OdFlow {
            origin: "A".to_string(),
            destination: "C".to_string(),
            flow: 10.0,
            edge_path: vec!["AB".to_string(), "BC".to_string()],
            cost: 2.0,
        })
        .collect();
    let failed_edges = vec!["AB".to_string()];
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
