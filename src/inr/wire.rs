use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Distribution};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::config::Config;

#[derive(Config)]
pub struct ComplexGaborConfig {
    pub in_features: usize,
    pub out_features: usize,
    #[config(default = true)]
    pub bias: bool,
    #[config(default = false)]
    pub is_first: bool,
    #[config(default = 10.0)]
    pub omega_0: f32,
    #[config(default = 40.0)]
    pub sigma_0: f32,
    #[config(default = false)]
    pub trainable: bool,
}

#[derive(Config)]
pub struct WireConfig {
    pub in_features: usize,
    pub hidden_features: usize,
    pub hidden_layers: usize,
    pub out_features: usize,
    #[config(default = true)]
    pub outermost_linear: bool,
    #[config(default = 30.0)]
    pub first_omega_0: f32,
    #[config(default = 30.0)]
    pub hidden_omega_0: f32,
    #[config(default = 10.0)]
    pub scale: f32,
}

/// Represents a complex tensor as two real tensors
#[derive(Module, Debug)]
pub struct ComplexTensor<B: Backend> {
    real: Tensor<B, 2>,
    imag: Tensor<B, 2>
}

impl<B: Backend> ComplexTensor<B> {
    pub fn new(real: Tensor<B, 2>, imag: Tensor<B, 2>) -> Self {
        Self { real, imag }
    }

    pub fn zeros(shape: [usize; 2], device: &B::Device) -> Self {
        Self {
            real: Tensor::zeros(shape, device),
            imag: Tensor::zeros(shape, device)
        }
    }

    pub fn complex_exp(&self) -> Self {
        let clamped_real = self.real.clone().clamp_min(-80.0).clamp_max(80.0);
        
        let exp_real = (-clamped_real).exp();
        
        let clamped_imag = self.imag.clone().clamp_min(-80.0).clamp_max(80.0);
        let sin_imag = clamped_imag.clone().sin();
        let cos_imag = clamped_imag.cos();
        
        Self {
            real: exp_real.clone() * cos_imag,
            imag: exp_real * sin_imag
        }
    }

    pub fn real(&self) -> Tensor<B, 2> {
        self.real.clone()
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            real: self.real.clone() + other.real.clone(),
            imag: self.imag.clone() + other.imag.clone(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ComplexLinear<B: Backend> {
    real: Linear<B>,
    imag: Linear<B>,
}

impl<B: Backend> ComplexLinear<B> {
    pub fn new(in_features: usize, out_features: usize, device: &B::Device, bias: bool) -> Self {
        Self {
            real: LinearConfig::new(in_features, out_features)
                .with_bias(bias)
                .init(device),
            imag: LinearConfig::new(in_features, out_features)
                .with_bias(bias)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> ComplexTensor<B> {
        ComplexTensor::new(
            self.real.forward(input.clone()),
            self.imag.forward(input)
        )
    }
}

#[derive(Module, Debug)]
pub struct ComplexGaborLayer<B: Backend> {
    omega_0: Param<Tensor<B, 1>>,
    scale_0: Param<Tensor<B, 1>>,
    linear: ComplexLinear<B>,
}

impl<B: Backend> ComplexGaborLayer<B> {
    pub fn new(config: &ComplexGaborConfig, device: &B::Device) -> Self {
        let omega_tensor = if config.trainable {
            Tensor::ones([1], device) * config.omega_0
        } else {
            (Tensor::ones([1], device) * config.omega_0).detach()
        };

        let scale_tensor = if config.trainable {
            Tensor::ones([1], device) * config.sigma_0
        } else {
            (Tensor::ones([1], device) * config.sigma_0).detach()
        };

        Self {
            omega_0: Param::from_tensor(omega_tensor),
            scale_0: Param::from_tensor(scale_tensor),
            linear: ComplexLinear::new(
                config.in_features, 
                config.out_features,
                device,
                config.bias
            ),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> ComplexTensor<B> {
        let lin = self.linear.forward(input);
        let [batch_size, hidden_size] = lin.real.dims();
        
        let omega_expanded = self.omega_0.val()
            .reshape([1, 1])
            .repeat(&[batch_size, hidden_size])
            .clamp_min(-10.0)  // prevent extreme frequency values
            .clamp_max(10.0);
            
        let omega = ComplexTensor::new(
            Tensor::zeros_like(&lin.real),
            omega_expanded * lin.real.clone()
        );
        
        let scale_expanded = self.scale_0.val()
            .reshape([1, 1])
            .repeat(&[batch_size, hidden_size])
            .clamp_min(0.0)    // scale should be positive
            .clamp_max(10.0);  // prevent extreme scaling
            
        let scale = scale_expanded * lin.real;
        let scale_squared = -(scale.abs().clamp_max(10.0).powf_scalar(2.0));
        
        let scale_term = ComplexTensor::new(
            scale_squared.clone(),
            Tensor::zeros_like(&scale_squared)
        );
        
        omega.add(&scale_term).complex_exp()
    }
}

#[derive(Module, Debug)]
pub struct Wire<B: Backend> {
    pub layers: Vec<ComplexGaborLayer<B>>,
    pub final_linear: ComplexLinear<B>,
}

impl<B: Backend> Wire<B> {
    pub fn new(config: &WireConfig, device: &B::Device) -> Self {
        let hidden_features = config.hidden_features;
        let mut layers = Vec::new();

        layers.push(ComplexGaborLayer::new(
            &ComplexGaborConfig::new(2, hidden_features)
                .with_omega_0(config.first_omega_0)
                .with_sigma_0(config.scale)
                .with_is_first(true),
            device,
        ));

        for _ in 0..config.hidden_layers {
            layers.push(ComplexGaborLayer::new(
                &ComplexGaborConfig::new(hidden_features, hidden_features)
                    .with_omega_0(config.hidden_omega_0)
                    .with_sigma_0(config.scale),
                device,
            ));
        }

        let final_linear = ComplexLinear::new(
            hidden_features,
            3, // 3 for rgb output
            device,
            true
        );

        Self {
            layers,
            final_linear,
        }
    }

    pub fn forward(&self, coords: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, input_dim] = coords.dims();
        assert_eq!(input_dim, 2, "Input coordinates must be 2D points");

        let mut x = self.layers[0].forward(coords);
        
        for layer in self.layers.iter().skip(1) {
            let real_part = x.real().clamp_min(-10.0).clamp_max(10.0);
            x = layer.forward(real_part);
        }
        
        let output = self.final_linear.forward(x.real())
            .real()
            .reshape([batch_size, 3]);
            
        output
    }
}