use clap::{crate_description, crate_name, crate_version, Parser, Subcommand};

#[derive(Parser, Debug)]
#[clap(name = crate_name!(),
    version = crate_version!(),
    about = crate_description!(),
    rename_all = "kebab-case"
)]
struct TpuClientTestArgs {
    #[clap(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Test the TpuClient's slot estimation accuracy.
    SlotEstimationAccuracy {
        #[clap(
            short,
            long,
            default_value_t = 0.15,
            help = "Accuracy threshold for slot estimation"
        )]
        accuracy_threshold: f64,
        #[clap(short, long, default_value_t = 100, help = "Number of samples to take")]
        num_samples: usize,
    },
}

mod modes;
use modes::*;

#[tokio::main]
async fn main() {
    solana_logger::setup_with("solana_tpu_client_test=info");

    let args = TpuClientTestArgs::parse();

    match args.mode {
        Mode::SlotEstimationAccuracy {
            accuracy_threshold,
            num_samples,
        } => slot_estimation_accuracy::run(accuracy_threshold, num_samples).await,
    }
}
