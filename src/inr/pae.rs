use burn::module::Module;
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use burn::config::Config;
use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    BatchNorm, BatchNormConfig,
    PaddingConfig1d,
    Linear, LinearConfig,
};
use crate::fft::gaders_whisper::{fft_frequencies_device, hann_window_device, stfft, tensor_max_scalar, tensor_min_scalar, tensor_log10, all_zeros, reverse, _10pow};

#[derive(Module, Debug, Clone)]
pub struct Elu {
    alpha: f32,
}

impl Elu {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let positive_part = tensor_max_scalar(x.clone(), 0.0);
        let negative_part = (tensor_min_scalar(x, 0.0).exp() - 1.0) * self.alpha;
        positive_part + negative_part
    }
}

#[derive(Config)]
pub struct PAEConfig {
    pub input_channels: usize,
    pub embedding_channels: usize,
    pub time_range: usize,
    pub window: f32,
    #[config(default = 3)]
    pub intermediate_channels: usize,
}

#[derive(Module, Debug)]
pub struct PAE<B: Backend> {
    encoder: PAEEncoder<B>,
    decoder: PAEDecoder<B>,
    phase_net: PhaseNetwork<B>,
    input_channels: usize,
    embedding_channels: usize,
    time_range: usize,
    window: f32,
    freqs: Tensor<B, 1>,
    tpi: Tensor<B, 1>,
    args: Tensor<B, 1>,
}

#[derive(Module, Debug)]
struct PAEEncoder<B: Backend> {
    conv1: Conv1d<B>,
    norm1: BatchNorm<B, 3>,
    elu1: Elu,
    conv2: Conv1d<B>,
}

#[derive(Module, Debug)]
struct PAEDecoder<B: Backend> {
    deconv1: Conv1d<B>,
    norm1: BatchNorm<B, 3>,
    elu1: Elu,
    deconv2: Conv1d<B>,
}

#[derive(Module, Debug)]
struct PhaseNetwork<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> PAE<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch_size, _] = input.dims();

        let y = input.clone().reshape([batch_size, self.input_channels, self.time_range]);

        // enc
        let y = self.encoder.conv1.forward(y);
        let y = self.encoder.norm1.forward(y);
        let y = self.encoder.elu1.forward(y);
        let latent = self.encoder.conv2.forward(y.clone());

        let (frequencies, amplitudes, offsets) = self.analyze_signal(y.clone());

        // pred
        let phases = self.phase_net.forward(y, &self.tpi);

        // rec
        let reconstructed = self.reconstruct_signal(&frequencies, &amplitudes, &phases, &offsets);

        // dec
        let y = self.decoder.deconv1.forward(reconstructed);
        let y = self.decoder.norm1.forward(y);
        let y = self.decoder.elu1.forward(y);
        let y = self.decoder.deconv2.forward(y);

        let output = y.reshape([batch_size, self.input_channels * self.time_range]);

        let params = vec![phases, frequencies, amplitudes, offsets];
        (output, latent, params)
    }

    fn analyze_signal(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, channels, _] = x.dims();
        
        // Compute STFT
        let (real, imag) = stfft(
            x.clone().reshape([batch * channels, self.time_range]),
            self.time_range,
            self.time_range,
            hann_window_device(self.time_range, &x.device())
        );
        
        let real = real.reshape([batch, channels, self.time_range / 2 + 1]);
        let imag = imag.reshape([batch, channels, self.time_range / 2 + 1]);

        let magnitudes = real.clone().powf_scalar(2.0) + imag.powf_scalar(2.0);
        let power = magnitudes.slice([0..0, 0..0, 1..self.time_range / 2 + 1]);

        let freqs = self.freqs.clone()
            .unsqueeze::<3>()
            .transpose();

        let freq_num = (power.clone() * freqs).sum_dim(2).squeeze(2);
        let freq_denom = power.clone().sum_dim(2).squeeze(2);
        let frequencies = freq_num / freq_denom;
    
        let amplitudes = (power.sum_dim(2).squeeze(2).sqrt() * 2.0) / (self.time_range as f32);
    
        let offsets = real.slice([0..0, 0..0, 0..1]).squeeze(2) / (self.time_range as f32);
    
        (frequencies, amplitudes, offsets)
    }

    fn reconstruct_signal(
        &self,
        frequencies: &Tensor<B, 2>,
        amplitudes: &Tensor<B, 2>,
        phases: &Tensor<B, 2>,
        offsets: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let device = &frequencies.device();
        let t = Tensor::arange(0..self.time_range as i64, device)
            .float()
            .reshape([1, 1, self.time_range])
            .repeat(&[amplitudes.dims()[0], self.embedding_channels, 1]);
    
        let angle = self.tpi.clone()
            .unsqueeze::<3>()
            .transpose()
            .mul(frequencies.clone().unsqueeze::<3>() * t + phases.clone().unsqueeze::<3>());
        
        amplitudes.clone().unsqueeze::<3>() * angle.sin() + offsets.clone().unsqueeze::<3>()
    }
}

impl PAEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PAE<B> {
        // Precompute constants
        let tpi = Tensor::from_floats([2.0 * std::f32::consts::PI], device);
        let args = linspace(-self.window / 2.0, self.window / 2.0, self.time_range, device);

        // Precompute FFT frequencies (excluding DC)
        let n = self.time_range;
        let k = Tensor::arange(1..(n / 2 + 1) as i64, device).float();
        let freqs = k * (n as f32 / self.window);

        let encoder = PAEEncoder {
            conv1: Conv1dConfig::new(self.input_channels, self.intermediate_channels, self.time_range)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            norm1: BatchNormConfig::new(self.intermediate_channels).init(device),
            elu1: Elu::new(1.0),
            conv2: Conv1dConfig::new(self.intermediate_channels, self.embedding_channels, self.time_range)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
        };

        let decoder = PAEDecoder {
            deconv1: Conv1dConfig::new(self.embedding_channels, self.intermediate_channels, self.time_range)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            norm1: BatchNormConfig::new(self.intermediate_channels).init(device),
            elu1: Elu::new(1.0),
            deconv2: Conv1dConfig::new(self.intermediate_channels, self.input_channels, self.time_range)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
        };

        let phase_layers = (0..self.embedding_channels)
            .map(|_| LinearConfig::new(self.time_range, 2).init(device))
            .collect();

        PAE {
            encoder,
            decoder,
            phase_net: PhaseNetwork { layers: phase_layers },
            input_channels: self.input_channels,
            embedding_channels: self.embedding_channels,
            time_range: self.time_range,
            window: self.window,
            freqs,
            tpi,
            args,
        }
    }
}

impl<B: Backend> PhaseNetwork<B> {
    fn forward(&self, x: Tensor<B, 3>, tpi: &Tensor<B, 1>) -> Tensor<B, 2> {
        let [batch_size, embedding_channels, time_range] = x.dims();
        let mut phases = Tensor::zeros([batch_size, embedding_channels], &x.device());

        for i in 0..embedding_channels {
            let channel_data = x.clone()
                .slice([0..batch_size, i..i+1, 0..time_range])
                .squeeze(1);
                
            let output = self.layers[i].forward(channel_data);
            let phase = atan2(
                output.clone().slice([0..batch_size, 1..2]),
                output.slice([0..batch_size, 0..1])
            );
            
            phases = phases.slice_assign(
                [0..batch_size, i..i+1], 
                phase / tpi.clone().unsqueeze()
            );
        }

        phases
    }
}

pub fn linspace<B: Backend>(start: f32, end: f32, steps: usize, device: &B::Device) -> Tensor<B, 1> {
    let step_size = (end - start) / (steps - 1) as f32;
    Tensor::arange(0..steps as i64, device)
        .float()
        .mul_scalar(step_size as f64)
        .add_scalar(start as f64)
}


  pub fn arctan<B: Backend>(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
    // Taylor series expansion of arctan(x)
    // arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ... (for |x| <= 1)
    // normalize input
    let scale = x.clone().abs().max();
    let x_norm = x.clone() / scale.clone();
    
    let x2 = x_norm.clone().powf_scalar(2.0);
    let x3 = x_norm.clone() * x2.clone();
    let x5 = x3.clone() * x2.clone();
    let x7 = x5.clone() * x2;
    
    let result = x_norm - (x3 / 3.0) + (x5 / 5.0) - (x7 / 7.0);
    result * scale
}

 fn atan2<B: Backend>(y: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = y.device();
    let pi = Tensor::from_floats([std::f32::consts::PI], &device);
    
    let quotient = y.clone() / x.clone();
    let mut angle = arctan(quotient);
    
    let mask_neg_x = x.clone().lower_elem(0.0); // x < 0
    let mask_zero_x = x.equal_elem(0.0); // x == 0
    let mask_pos_y = y.clone().greater_elem(0.0); // y > 0
    let mask_neg_y = y.clone().lower_elem(0.0); // y < 0
    
    let adjustment = mask_neg_x.float() * pi.clone() * (mask_pos_y.float() - mask_neg_y.float());
    let vertical = mask_zero_x.clone().float() * (pi.clone() / 2.0) * y.clone().sign();
    
    angle = angle + adjustment + vertical;
    
    let zero_mask = mask_zero_x.float() * y.equal_elem(0.0).float();
    angle = angle.mul(-zero_mask + 1.0);
    
    ngle
    }