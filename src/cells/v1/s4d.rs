// errors in variable dimensions

// use burn::tensor::backend::Backend;
// use burn::tensor::{Tensor, Distribution, ElementConversion};
// use burn::module::{Module, Param};
// use burn::nn::{Dropout, DropoutConfig, Gelu};
// use burn::config::Config;
// use std::f32::consts::PI;
// use thiserror::Error;
// use burn::prelude::Int;
// use crate::fft::BatchedFft;

// #[derive(Error, Debug)]
// pub enum S4DError {
//     #[error("Invalid input dimensions: expected [..., {0}], got [..., {1}]")]
//     DimensionMismatch(usize, usize),
//     #[error("Invalid state size: must be even for complex number handling")]
//     InvalidStateSize,
//     #[error("FFT error: {0}")]
//     FftError(#[from] crate::fft::FftError),
// }

// #[derive(Config)]
// pub struct S4DKernelConfig {
//     pub d_model: usize,
//     pub n_state: usize,
//     #[config(default = 0.001)]
//     pub dt_min: f32,
//     #[config(default = 0.1)]
//     pub dt_max: f32,
// }

// #[derive(Module, Debug)]
// pub struct S4DKernel<B: Backend> {
//     c_re: Param<Tensor<B, 2>>,
//     c_im: Param<Tensor<B, 2>>,
//     log_dt: Param<Tensor<B, 1>>,
//     log_a_real: Param<Tensor<B, 2>>,
//     a_imag: Param<Tensor<B, 2>>,
// }

// impl<B: Backend> S4DKernel<B> {
//     pub fn init(config: S4DKernelConfig, device: &B::Device) -> Result<Self, S4DError> {
//         if config.n_state % 2 != 0 {
//             return Err(S4DError::InvalidStateSize);
//         }

//         let h = config.d_model;
//         let n = config.n_state / 2;

//         // Initialize parameters with explicit shapes
//         let c_re = Tensor::random([h, n], Distribution::Normal(0.0, 1.0), device);
//         let c_im = Tensor::random([h, n], Distribution::Normal(0.0, 1.0), device);
//         let log_dt = Tensor::random(
//             [h], 
//             Distribution::Uniform(config.dt_min.ln() as f64, config.dt_max.ln() as f64),
//             device
//         );
//         let log_a_real = Tensor::full([h, n], (0.5f32).ln(), device);
        
//         // Create a_imag with broadcasting
//         let a_imag_base = Tensor::<B, 1, Int>::arange(0..n as i64, device);
//         let a_imag_float = a_imag_base.float() * PI;
//         let a_imag = a_imag_float.reshape([1, n]).repeat(&[h, 1]);

//         Ok(Self {
//             c_re: Param::from_tensor(c_re),
//             c_im: Param::from_tensor(c_im),
//             log_dt: Param::from_tensor(log_dt),
//             log_a_real: Param::from_tensor(log_a_real),
//             a_imag: Param::from_tensor(a_imag),
//         })
//     }

//     pub fn forward(&self, seq_len: usize) -> Tensor<B, 3> {
//         let device = self.c_re.val().device();
//         let [h, n] = self.c_re.val().dims();
        
//         // Get real parameters
//         let dt = self.log_dt.val().exp().reshape([h, 1]);
//         let c_re = self.c_re.val();
//         let c_im = self.c_im.val();
//         let a_re = -self.log_a_real.val().exp();
//         let a_im = self.a_imag.val();

//         // Compute dtA 
//         let dta_re = a_re.clone() * dt.clone();
//         let dta_im = a_im.clone() * dt;

//         // Create time steps tensor
//         let time_steps = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).float();
//         let time_steps = time_steps.reshape([1, seq_len]).repeat(&[n, 1]); // [n, l]

//         // Compute kernel in frequency domain
//         let k_re = dta_re.transpose().reshape([n, 1]) * time_steps.clone();
//         let k_im = dta_im.transpose().reshape([n, 1]) * time_steps;

//         let exp_k_re = (-k_re.clone()).exp() * k_im.clone().cos();
//         let exp_k_im = (-k_re).exp() * k_im.sin();

//         // Scale by C with proper broadcasting
//         let denominator = (a_re.powf_scalar(2.0) + a_im.powf_scalar(2.0))
//             .sqrt()
//             .transpose(); // [n, h]
        
//         let scaling = (exp_k_re.clone() - 1.0) / denominator.reshape([n, 1]);

//         let c_re = c_re.transpose(); // [n, h]
//         let c_im = c_im.transpose(); // [n, h]

//         // Final computation with proper broadcasting
//         let k_re = (c_re * scaling.clone() * exp_k_re - c_im * scaling * exp_k_im)
//             .reshape([1, h, seq_len]);
        
//         k_re * 2.0
//     }
// }

// #[derive(Config)]
// pub struct S4DConfig {
//     pub d_model: usize,
//     pub d_state: usize,
//     #[config(default = 0.0)]
//     pub dropout: f64,
//     #[config(default = true)]
//     pub transposed: bool,
//     pub kernel: S4DKernelConfig,
// }

// impl S4DConfig {
//     /// Initialize a new S4D model from this configuration
//     pub fn init<B: Backend>(&self, device: &B::Device) -> Result<S4D<B>, S4DError> {
//         S4D::new(self.clone(), device)
//     }
// }

// #[derive(Module, Debug)]
// pub struct S4D<B: Backend> {
//     kernel: S4DKernel<B>,
//     d: Param<Tensor<B, 1>>,
//     dropout: Dropout,
//     output_linear: burn::nn::conv::Conv1d<B>,
//     activation: Gelu,
//     transposed: bool,
// }

// impl<B: Backend> S4D<B> {
//     pub fn new(config: S4DConfig, device: &B::Device) -> Result<Self, S4DError> {
//         let kernel = S4DKernel::init(config.kernel, device)?;
        
//         let d = Tensor::random([config.d_model], Distribution::Normal(0.0, 1.0), device);
//         let dropout = DropoutConfig::new(config.dropout).init();
        
//         // Fixed Conv1d configuration
//         let output_linear = burn::nn::conv::Conv1dConfig::new(
//             config.d_model,  // in_channels
//             config.d_model,  // out_channels
//             1               // kernel_size
//         )
//         .with_bias(true)
//         .init(device);

//         Ok(Self {
//             kernel,
//             d: Param::from_tensor(d),
//             dropout,
//             output_linear,
//             activation: Gelu::new(),
//             transposed: config.transposed,
//         })
//     }

//     pub fn forward(&self, input: Tensor<B, 3>) -> Result<(Tensor<B, 3>, Option<Tensor<B, 3>>), S4DError> {
//         // Handle transposition if needed
//         let u = if self.transposed {
//             input
//         } else {
//             input.movedim(1, 2)
//         };
    
//         let [batch_size, hidden_size, seq_len] = u.dims();
    
//         // Compute SSM Kernel and broadcast
//         let k = self.kernel.forward(seq_len); // [1, hidden_size, seq_len]
//         let k = k.repeat(&[batch_size, 1, 1]); // [batch_size, hidden_size, seq_len]
    
//         // Prepare for FFT: Both tensors are [batch_size, hidden_size, seq_len]
//         let device = u.device().clone(); // Take ownership of device
//         let fft = BatchedFft::new(seq_len * 2, device);
    
//         // Reshape maintaining batch size: [batch_size * hidden_size, seq_len]
//         let u_flat = u.clone().reshape([batch_size * hidden_size, seq_len]);
//         let k_flat = k.reshape([batch_size * hidden_size, seq_len]);
    
//         // Perform FFT
//         let (u_f_re, u_f_im) = fft.rfft_batched(u_flat)?;
//         let (k_f_re, k_f_im) = fft.rfft_batched(k_flat)?;
    
//         // Complex multiplication in frequency domain
//         let y_f_re = u_f_re.clone() * k_f_re.clone() - u_f_im.clone() * k_f_im.clone();
//         let y_f_im = u_f_re * k_f_im + u_f_im * k_f_re;
    
//         // Inverse FFT
//         let y_flat = fft.irfft_batched(y_f_re, y_f_im, seq_len)?;
    
//         // Reshape back to 3D: [batch_size, hidden_size, seq_len]
//         let mut y = y_flat.reshape([batch_size, hidden_size, seq_len]);
    
//         // Add skip connection with proper broadcasting
//         let d_broadcast = self.d.val()
//             .reshape([1, hidden_size, 1])
//             .repeat(&[batch_size, 1, seq_len]);
        
//         y = y + u * d_broadcast;
    
//         // Apply dropout and activation while maintaining 3D shape
//         y = self.dropout.forward(self.activation.forward(y));
    
//         // Apply output projection - Conv1d expects [batch_size, channels, length]
//         y = self.output_linear.forward(y);
    
//         // Handle transpose back if needed
//         let output = if self.transposed {
//             y
//         } else {
//             y.movedim(2, 1)
//         };
    
//         Ok((output, None))
//     }
// }