/*
    To run this benchmark:
    `cargo bench --bench repair_signing`
*/

use {
    criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main},
    solana_core::repair::serve_repair::{
        RepairProtocol, RepairRequestHeader, RepairSigningPool, ServeRepair,
    },
    solana_keypair::{Keypair, Signer},
    solana_pubkey::Pubkey,
    std::{hint::black_box, net::SocketAddr, num::NonZeroUsize, sync::Arc},
};

const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 512];
const POOL_BATCH_SIZES: &[usize] = &[4, 8, 16, 32, 64, 128, 512];
const WORKER_COUNTS: &[usize] = &[4, 8, 12, 16];
const MIN_SIGNATURES_PER_WORKER: &[usize] = &[2, 4, 8, 16];

fn make_repair_protocol(index: usize, sender: Pubkey, recipient: Pubkey) -> RepairProtocol {
    RepairProtocol::WindowIndex {
        header: RepairRequestHeader::new(sender, recipient, 0, index as u32),
        slot: 42,
        shred_index: index as u64,
    }
}

fn make_repair_batch(
    batch_size: usize,
    sender: Pubkey,
    recipient: Pubkey,
) -> Vec<(SocketAddr, RepairProtocol)> {
    let addr = "127.0.0.1:1234".parse().unwrap();
    (0..batch_size)
        .map(|index| (addr, make_repair_protocol(index, sender, recipient)))
        .collect()
}

fn bench_repair_proto_to_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("repair_proto_to_bytes");
    let keypair = Keypair::new();
    let recipient = Pubkey::new_unique();
    let request = make_repair_protocol(0, keypair.pubkey(), recipient);

    group.bench_function("window_index", |b| {
        b.iter(|| {
            let bytes =
                ServeRepair::repair_proto_to_bytes(black_box(&request), black_box(&keypair))
                    .unwrap();
            black_box(bytes);
        });
    });
    group.finish();
}

fn bench_repair_signing_raw_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("repair_signing_raw_loop");
    let keypair = Arc::new(Keypair::new());
    let recipient = Pubkey::new_unique();
    let sender = keypair.pubkey();

    for &batch_size in BATCH_SIZES {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter_batched(
                    || make_repair_batch(batch_size, sender, recipient),
                    |batch| {
                        let mut num_signed = 0;
                        for (_, request) in batch {
                            if let Ok(bytes) =
                                ServeRepair::repair_proto_to_bytes(&request, &keypair)
                            {
                                black_box(bytes);
                                num_signed += 1;
                            }
                        }
                        black_box(num_signed);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_repair_signing_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("repair_signing_pool");
    let keypair = Arc::new(Keypair::new());
    let recipient = Pubkey::new_unique();
    let sender = keypair.pubkey();

    for &num_workers in WORKER_COUNTS {
        for &min_signatures_per_worker in MIN_SIGNATURES_PER_WORKER {
            let mut pool = RepairSigningPool::new_with_min_signatures_per_worker(
                NonZeroUsize::new(num_workers).unwrap(),
                NonZeroUsize::new(min_signatures_per_worker).unwrap(),
            );

            // Warm the worker threads and scratch buffers before Criterion starts timing.
            let warmup_batch = make_repair_batch(512, sender, recipient);
            black_box(pool.sign_batch(keypair.clone(), warmup_batch));

            for &batch_size in POOL_BATCH_SIZES {
                let id = BenchmarkId::new(
                    format!("workers={num_workers}/min={min_signatures_per_worker}"),
                    batch_size,
                );
                group.bench_with_input(id, &batch_size, |b, _| {
                    b.iter_batched(
                        || make_repair_batch(batch_size, sender, recipient),
                        |batch| {
                            let signed = pool.sign_batch(keypair.clone(), batch);
                            black_box(signed.len());
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_repair_proto_to_bytes,
    bench_repair_signing_raw_loop,
    bench_repair_signing_pool
);
criterion_main!(benches);
