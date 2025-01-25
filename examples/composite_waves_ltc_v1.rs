use burn::{
    backend::Autodiff,
    config::Config,
    data::{
        dataloader::{
            batcher::Batcher, 
            DataLoaderBuilder
        },
        dataset::Dataset,
    },
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{
            Backend, AutodiffBackend
        }, 
        Tensor, 
        TensorData
    },
    train::{
        metric::LossMetric,
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use burn_tch::{
    LibTorch, LibTorchDevice
};

use plotters::prelude::*;
use std::f32::consts::PI;
use rand::Rng;
use uuid::Uuid;

use synapse::{
    cells::{
        v1::ltc::{
            LTCConfig, LTC
        },
    },
    wiring::v1::{
        base::WiringImpl, 
        auto_ncp::AutoNCP,
        // fully_connected::FullyConnected
    },
};

fn plot_results(input: &Vec<Vec<f32>>, true_output: &Vec<f32>, model_output: &Vec<f32>, filename: &str) {
    let root_area = BitMapBackend::new(filename, (1024, 768))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Wave Prediction vs True Output", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..(input.len() + 10), -2.0f32..2.0f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let num_waves = input[0].len();
    let sequence_length = input.len();
    let mut transposed_input: Vec<Vec<f32>> = vec![vec![0.0; sequence_length]; num_waves];

    for t in 0..sequence_length {
        for w in 0..num_waves {
            transposed_input[w][t] = input[t][w];
        }
    }

    for (wave_idx, wave_values) in transposed_input.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                (0..).zip(wave_values.iter().cloned()),
                Palette99::pick(wave_idx).mix(0.5),
            ))
            .unwrap()
            .label(format!("Input Wave {}", wave_idx + 1))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(wave_idx).mix(0.5))
            });
    }

    let true_output_style = ShapeStyle {
        color: GREEN.to_rgba(),
        filled: false,
        stroke_width: 3,
    };
    
    let model_output_style = ShapeStyle {
        color: BLUE.to_rgba(),
        filled: false,
        stroke_width: 3,
    };

    chart
        .draw_series(LineSeries::new(
            (0..).zip(true_output.iter().cloned()),
            true_output_style,
        ))
        .unwrap()
        .label("True Output")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.filled()));

    chart
        .draw_series(LineSeries::new(
            (0..).zip(model_output.iter().cloned()),
            model_output_style,
        ))
        .unwrap()
        .label("Model Output")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

#[derive(Clone)]
struct WaveDataset {
    num_samples: usize,
    num_waves: usize,
    sequence_length: usize,
}

impl WaveDataset {
    fn new(num_samples: usize, num_waves: usize, sequence_length: usize) -> Self {
        Self {
            num_samples,
            num_waves,
            sequence_length,
        }
    }

    fn generate_sequence(&self) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rng = rand::thread_rng();

        let wave_params: Vec<(f32, f32, f32)> = (0..self.num_waves)
            .map(|_| {
                let amplitude = rng.gen_range(0.005..0.12);
                let frequency = rng.gen_range(2.0..10.0);
                let phase = rng.gen_range(0.0..2.0 * PI);
                (amplitude, frequency, phase)
            })
            .collect();

        let mut inputs = Vec::with_capacity(self.sequence_length);
        let mut outputs = Vec::with_capacity(self.sequence_length);

        for t in 0..self.sequence_length {
            let t = t as f32 / self.sequence_length as f32;

            let waves: Vec<f32> = wave_params
                .iter()
                .map(|(amp, freq, phase)| amp * (2.0 * PI * freq * t + phase).sin())
                .collect();

            let sum = waves.iter().sum();

            inputs.push(waves);
            outputs.push(sum);
        }

        (inputs, outputs)
    }
}

impl Dataset<(Vec<Vec<f32>>, Vec<f32>)> for WaveDataset {
    fn get(&self, index: usize) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        if index >= self.num_samples {
            return None;
        }
        Some(self.generate_sequence())
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

#[derive(Clone)]
struct WaveBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct WaveBatch<B: Backend> {
    waves: Tensor<B, 3>, // [batch_size, sequence_length, num_waves]
    sums: Tensor<B, 2>,  // [batch_size, sequence_length]
}

impl<B: Backend> WaveBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<(Vec<Vec<f32>>, Vec<f32>), WaveBatch<B>> for WaveBatcher<B> {
    fn batch(&self, items: Vec<(Vec<Vec<f32>>, Vec<f32>)>) -> WaveBatch<B> {
        let batch_size = items.len();
        let sequence_length = items[0].0.len();
        let num_waves = items[0].0[0].len();

        let waves: Vec<f32> = items
            .iter()
            .flat_map(|(waves, _)| waves.iter().flatten().copied())
            .collect();
        let waves_data = TensorData::new(waves, vec![batch_size, sequence_length, num_waves]);
        let waves = Tensor::from_data(waves_data, &self.device);

        let sums: Vec<f32> = items
            .iter()
            .flat_map(|(_, sums)| sums.iter().copied())
            .collect();
        let sums_data = TensorData::new(sums, vec![batch_size, sequence_length]);
        let sums = Tensor::from_data(sums_data, &self.device);

        WaveBatch { waves, sums }
    }
}

#[derive(Module, Debug)]
struct WaveModel<B: Backend> {
    ltc: LTC<B>,
}

impl<B: Backend> WaveModel<B> {
    pub fn new(mut wiring: WiringImpl<B>, num_waves: usize, hidden_size: usize, ltc_config: LTCConfig, model_name: &str, uuid1: &str, device: &B::Device) -> Self {

        wiring.build(num_waves).unwrap();

        let filename = format!("./output/{}_{}_{}/{}_{}_{}_{}.png", model_name,ltc_config.seed, uuid1, wiring.to_string(), hidden_size, ltc_config.clone().integration_method, ltc_config.clone().ode_unfolds);
        wiring.draw_graph(&filename).unwrap();

        let ltc = LTC::new(ltc_config, device, wiring);

        Self { ltc }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (output, _) = self.ltc.forward(x, None, None);
        output.squeeze(2)
    }
}

impl<B: AutodiffBackend> TrainStep<WaveBatch<B>, RegressionOutput<B>> for WaveModel<B> {
    fn step(&self, batch: WaveBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward(batch.waves);

        let diff = output.clone() - batch.sums.clone();
        let loss = diff.clone() * diff;
        let loss = loss.mul_scalar(10.0);
        let loss = loss.mean();

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput::new(loss, output, batch.sums),
        )
    }
}

impl<B: Backend> ValidStep<WaveBatch<B>, RegressionOutput<B>> for WaveModel<B> {
    fn step(&self, batch: WaveBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.waves);

        let diff = output.clone() - batch.sums.clone();
        let loss = diff.clone() * diff;
        let loss = loss.mul_scalar(10.0);
        let loss = loss.mean();

        RegressionOutput::new(loss, output, batch.sums)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model_hidden_size: usize,
    pub optimizer: AdamConfig,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

#[derive(Config)]
pub struct DatasetConfig {
    #[config(default = 4)]
    pub num_waves: usize,
    #[config(default = 100)]
    pub sequence_length: usize,
}

pub fn train<B: AutodiffBackend>(wiring: WiringImpl<B>, config: TrainingConfig, ltc_config: LTCConfig, dataset_config: DatasetConfig, device: B::Device) {

    let num_waves = dataset_config.num_waves;
    let sequence_length = dataset_config.sequence_length;

    let train_dataset = WaveDataset::new(1000, num_waves, sequence_length); // 1000
    let valid_dataset = WaveDataset::new(200, num_waves, sequence_length); // 200

    let batcher_train = WaveBatcher::<B>::new(device.clone());
    let batcher_valid = WaveBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let model_name = "wave_model";
    let uuid1 = Uuid::new_v4().to_string();
    let output_directory = format!(
        "./output/{}_{}_{}", 
        model_name,
        ltc_config.seed,
        uuid1
    );
    std::fs::create_dir_all(output_directory.clone()).expect("Failed to create output directory");

    let model = WaveModel::new(wiring, num_waves, config.model_hidden_size, ltc_config.clone(), &model_name, &uuid1, &device);

    let learner = LearnerBuilder::new(output_directory)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    for _ in 0..10 {
        let test_dataset = WaveDataset::new(1, num_waves, sequence_length);
        if let Some((input, true_output)) = test_dataset.get(0) {
            let input_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(
                    input.iter().flatten().cloned().collect(),
                    vec![1, sequence_length, num_waves]
                ),
                &device,
            );
            let model_output_tensor = trained_model.forward(input_tensor);
            let model_output: Vec<f32> = model_output_tensor.to_data().to_vec().expect("Failed to get output data");

            let uuid2 = Uuid::new_v4().to_string();

            let filename = format!(
                "./output/{}_{}_{}/test_sample_{}.png", 
                model_name,
                ltc_config.seed,
                uuid1,
                uuid2
            );


            plot_results(&input, &true_output, &model_output, &filename);
            println!("Plot saved to {}", filename);
        }
    }
}

fn main() {
    let device = LibTorchDevice::Mps;

    println!("Using {:?} backend", device);

    let hidden_size = 8; //12
    let model_seed = 12;

    let dataset_config = DatasetConfig {
        num_waves: 4,
        sequence_length: 100,
    };

    let wiring = WiringImpl::AutoNCP(
        AutoNCP::new(
            hidden_size,      // Total number of neurons
            1,                // Output size (1 for sum prediction)
            device.clone(),
            Some(0.9),        // Sparsity level (0.1 to 0.9)
            Some(model_seed),         // Random seed //42
        )
        .unwrap()
    );

    let train_config = TrainingConfig {
        model_hidden_size: hidden_size, //12
        optimizer: AdamConfig::new(),
        num_epochs: 6,
        batch_size: 10,
        num_workers: 4,
        seed: 42,
        learning_rate: 1e-3,
    };

    let ltc_config = LTCConfig {
        input_size: dataset_config.clone().num_waves,
        return_sequences: true,
        batch_first: true,
        mixed_memory: true,
        input_mapping: "affine".into(),
        output_mapping: "affine".into(),
        ode_unfolds: 3,
        epsilon: 1e-10,
        implicit_param_constraints: true,
        integration_method: "euler".into(), // "euler".into(), // "cash_karp".into(), // "backward_euler".into(),
        tolerance: 1e-4,
        seed: model_seed
    };

    train::<Autodiff<LibTorch>>(wiring, train_config, ltc_config, dataset_config, device);
}