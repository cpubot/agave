use {
    log::info,
    solana_client::{connection_cache::Protocol, nonblocking::tpu_client::LeaderTpuService},
    solana_sdk::clock::DEFAULT_MS_PER_SLOT,
    solana_test_validator::{TestValidator, TestValidatorGenesis},
    std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    },
    tokio::time::sleep,
};

struct SlotEstimationAccuracy {
    leader_tpu_service: LeaderTpuService,
    validator: TestValidator,
    exit: Arc<AtomicBool>,
}

impl SlotEstimationAccuracy {
    async fn new() -> Self {
        let validator = TestValidatorGenesis::default().start_async().await.0;
        let rpc_client = Arc::new(validator.get_async_rpc_client());
        let exit = Arc::new(AtomicBool::new(false));
        let leader_tpu_service = LeaderTpuService::new(
            rpc_client,
            &validator.rpc_pubsub_url(),
            Protocol::QUIC,
            exit.clone(),
        )
        .await
        .unwrap();

        Self {
            leader_tpu_service,
            validator,
            exit,
        }
    }
}

pub(crate) async fn run(accuracy_threshold: f64, num_samples: usize) {
    info!("bootstrapping test validator");

    let SlotEstimationAccuracy {
        mut leader_tpu_service,
        validator,
        exit,
    } = SlotEstimationAccuracy::new().await;

    let sleep_time = Duration::from_millis(DEFAULT_MS_PER_SLOT);
    let mut result_pairs = vec![];

    let mut actual = validator.current_slot();
    while (actual as usize) < num_samples {
        actual = validator.current_slot();
        let estimated = leader_tpu_service.estimated_current_slot();
        result_pairs.push((estimated, actual));
        info!(
            "estimated: {}, actual: {} {}",
            estimated,
            actual,
            if estimated == actual { "✅" } else { "❌" }
        );

        sleep(sleep_time).await;
    }

    let failure_rate =
        result_pairs.iter().filter(|(a, b)| a != b).count() as f64 / result_pairs.len() as f64;

    info!(
        "failure rate: {failure_rate} <= {accuracy_threshold} {}",
        if failure_rate <= accuracy_threshold {
            "✅"
        } else {
            "❌"
        }
    );

    info!("cleaning up and exiting...");

    exit.store(true, Ordering::Relaxed);
    leader_tpu_service.join().await;
}
