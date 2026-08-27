use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use _core::arrow_ffi::{prepare_network_batches, read_network_batches};
use _core::core::{allocate, disrupt, shortest_path, Demand, Edge, OdFlow};

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

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

/// A link table in the shape Python hands across the FFI boundary.
///
/// The readers used to resolve each numeric column by name for every row;
/// this bench guards the per-batch resolution that replaced it.
fn synthetic_network_batch(size: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("edge_from", DataType::Int64, false),
        Field::new("edge_to", DataType::Int64, false),
        Field::new("edge_id", DataType::Int64, false),
        Field::new("cost", DataType::Float64, false),
        Field::new("capacity", DataType::Float64, true),
    ]);
    let from: Vec<i64> = (0..size as i64).collect();
    let to: Vec<i64> = (1..=size as i64).collect();
    let id: Vec<i64> = (0..size as i64).collect();
    let cost: Vec<f64> = (0..size).map(|index| 1.0 + (index % 7) as f64).collect();
    let capacity: Vec<f64> = vec![1_000.0; size];

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(from)),
            Arc::new(Int64Array::from(to)),
            Arc::new(Int64Array::from(id)),
            Arc::new(Float64Array::from(cost)),
            Arc::new(Float64Array::from(capacity)),
        ],
    )
    .expect("valid record batch")
}

fn bench_network_parsing(c: &mut Criterion) {
    let batches = vec![synthetic_network_batch(3_000)];
    c.bench_function("read_network_batches_3000_edges", |b| {
        b.iter(|| read_network_batches(black_box(&batches)).expect("readable"))
    });
    c.bench_function("prepare_network_batches_3000_edges", |b| {
        b.iter(|| prepare_network_batches(black_box(&batches)).expect("preparable"))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_shortest_path,
        bench_allocation,
        bench_capacity_allocation,
        bench_disruption,
        bench_network_parsing
}
criterion_main!(benches);
