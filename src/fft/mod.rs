use burn::tensor::{backend::Backend, Tensor};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FftError {
    #[error("Input size {0} is not valid for FFT operation")]
    InvalidInputSize(usize),
    #[error("Input dimensions do not match: real={0}, imag={1}")]
    DimensionMismatch(usize, usize),
    #[error("Output size {0} is not compatible with input size {1}")]
    InvalidOutputSize(usize, usize),
}

#[derive(Debug)]
pub struct TwiddleFactors<B: Backend> {
    cos_table: Tensor<B, 2>,
    sin_table: Tensor<B, 2>,
    max_size: usize,
}

impl<B: Backend> TwiddleFactors<B> {
    pub fn new(max_size: usize, device: B::Device) -> Self {
        println!("Creating twiddle table of size {}", max_size);
        let mut cos_table = Tensor::zeros([max_size/2 + 1, max_size], &device);
        let mut sin_table = Tensor::zeros([max_size/2 + 1, max_size], &device);

        for k in 0..=max_size/2 {
            for n in 0..max_size {
                let angle = -2.0 * PI * (k as f64) * (n as f64) / (max_size as f64);
                let cos_val = angle.cos() as f32;
                let sin_val = angle.sin() as f32;
                
                cos_table = cos_table.slice_assign(
                    [k..k+1, n..n+1],
                    Tensor::from_data([[cos_val]], &device)
                );
                sin_table = sin_table.slice_assign(
                    [k..k+1, n..n+1],
                    Tensor::from_data([[sin_val]], &device)
                );
            }
        }
        println!("Twiddle table created");

        Self {
            cos_table,
            sin_table,
            max_size,
        }
    }

    fn get_factors(&self, size: usize) -> Result<(Tensor<B, 2>, Tensor<B, 2>), FftError> {
        if size > self.max_size {
            return Err(FftError::InvalidInputSize(size));
        }
        
        Ok((
            self.cos_table.clone().slice([0..size/2 + 1, 0..size]),
            self.sin_table.clone().slice([0..size/2 + 1, 0..size])
        ))
    }
}

/// Cooley-Tukey
pub struct Fft<B: Backend> {
    twiddle_factors: TwiddleFactors<B>,
}

impl<B: Backend> Fft<B> {
    pub fn new(max_size: usize, device: B::Device) -> Self {
        Self {
            twiddle_factors: TwiddleFactors::new(max_size, device),
        }
    }

    pub fn rfft(&self, input: Tensor<B, 1>) -> Result<(Tensor<B, 1>, Tensor<B, 1>), FftError> {
        let n = input.dims()[0];
        if !n.is_power_of_two() {
            return Err(FftError::InvalidInputSize(n));
        }
    
        let output_size = n/2 + 1;
        let (cos_table, sin_table) = self.twiddle_factors.get_factors(n)?;
        
        let dc_component = input.clone().mean().reshape([1]);
        let mut real = Tensor::zeros([output_size], &input.device());
        let mut imag = Tensor::zeros([output_size], &input.device());
        
        real = real.slice_assign([0..1], dc_component);
    
        for k in 1..output_size {
            let cos_k = cos_table.clone().slice([k..k+1, 0..n]).reshape([n]);
            let sin_k = sin_table.clone().slice([k..k+1, 0..n]).reshape([n]);

            let real_k = (input.clone() * cos_k).sum().reshape([1]);
            let imag_k = (input.clone() * sin_k).sum().reshape([1]);
            
            real = real.slice_assign([k..k+1], real_k);
            imag = imag.slice_assign([k..k+1], imag_k);
        }
    
        let norm_factor = (n as f32).sqrt();
        real = real / norm_factor;
        imag = imag / norm_factor;
    
        Ok((real, imag))
    }
    
    pub fn irfft(
        &self,
        real: Tensor<B, 1>,
        imag: Tensor<B, 1>,
        output_size: usize
    ) -> Result<Tensor<B, 1>, FftError> {
        let freq_size = real.dims()[0];
        if freq_size != imag.dims()[0] {
            return Err(FftError::DimensionMismatch(freq_size, imag.dims()[0]));
        }
        if output_size < 2 * (freq_size - 1) {
            return Err(FftError::InvalidOutputSize(output_size, freq_size));
        }
    
        let (cos_table, sin_table) = self.twiddle_factors.get_factors(output_size)?;
        let mut output = Tensor::zeros([output_size], &real.device());
    
        let dc = real.clone().slice([0..1]);
    
        for n in 0..output_size {
            let mut value = dc.clone();
            
            for k in 1..freq_size {
                let cos_nk = cos_table.clone().slice([k..k+1, n..n+1]).reshape([1]);
                let sin_nk = sin_table.clone().slice([k..k+1, n..n+1]).reshape([1]);
                
                let real_k = real.clone().slice([k..k+1]);
                let imag_k = imag.clone().slice([k..k+1]);
        
                let contribution = (real_k * cos_nk + imag_k * sin_nk) * 2.0;
                value = value + contribution;
            }
            
            output = output.slice_assign([n..n+1], value);
        }
    
        output = output / (output_size as f32).sqrt();
    
        Ok(output)
    }

    pub fn pad_to_power_of_two(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let current_size = input.dims()[0];
        let target_size = current_size.next_power_of_two();
        
        if current_size == target_size {
            return input;
        }

        let mut padded = Tensor::zeros([target_size], &input.device());
        padded = padded.slice_assign([0..current_size], input);
        padded
    }

    pub fn apply_hann_window(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let n = input.dims()[0];
        let mut window = Tensor::zeros([n], &input.device());
        
        for i in 0..n {
            let value = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()) as f32;
            window = window.slice_assign(
                [i..i+1],
                Tensor::from_data([value], &input.device())
            );
        }

        input * window
    }
}

pub struct BatchedFft<B: Backend> {
    fft: Fft<B>,
}

impl<B: Backend> BatchedFft<B> {
    pub fn new(max_size: usize, device: B::Device) -> Self {
        Self {
            fft: Fft::new(max_size, device),
        }
    }

    pub fn rfft_batched(&self, input: Tensor<B, 2>) -> Result<(Tensor<B, 2>, Tensor<B, 2>), FftError> {
        let [batch_size, signal_length] = input.dims();
        let output_size = signal_length/2 + 1;
        
        let mut real_out = Tensor::zeros([batch_size, output_size], &input.device());
        let mut imag_out = Tensor::zeros([batch_size, output_size], &input.device());

        for b in 0..batch_size {
            let input_slice = input.clone().slice([b..b+1, 0..signal_length]).squeeze_dims(&[0]);
            let (real, imag) = self.fft.rfft(input_slice)?;
            
            real_out = real_out.slice_assign([b..b+1, 0..output_size], real.unsqueeze_dim(0));
            imag_out = imag_out.slice_assign([b..b+1, 0..output_size], imag.unsqueeze_dim(0));
        }

        Ok((real_out, imag_out))
    }

    pub fn irfft_batched(
        &self,
        real: Tensor<B, 2>,
        imag: Tensor<B, 2>,
        output_size: usize
    ) -> Result<Tensor<B, 2>, FftError> {
        let [batch_size, freq_size] = real.dims();
        
        let mut output = Tensor::zeros([batch_size, output_size], &real.device());

        for b in 0..batch_size {
            let real_slice = real.clone().slice([b..b+1, 0..freq_size]).squeeze_dims(&[0]);
            let imag_slice = imag.clone().slice([b..b+1, 0..freq_size]).squeeze_dims(&[0]);
            
            let reconstructed = self.fft.irfft(real_slice, imag_slice, output_size)?;
            output = output.slice_assign([b..b+1, 0..output_size], reconstructed.unsqueeze_dim(0));
        }

        Ok(output)
    }
}
