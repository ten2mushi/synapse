// src/flow_matching/scheduler.rs

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct SchedulerOutput<B: Backend> {
    pub alpha_t: Tensor<B, 1>,
    pub sigma_t: Tensor<B, 1>,
    pub d_alpha_t: Tensor<B, 1>,
    pub d_sigma_t: Tensor<B, 1>,
}

impl<B: Backend> SchedulerOutput<B> {
    pub fn new(
        alpha_t: Tensor<B, 1>,
        sigma_t: Tensor<B, 1>,
        d_alpha_t: Tensor<B, 1>,
        d_sigma_t: Tensor<B, 1>,
    ) -> Self {
        Self {
            alpha_t,
            sigma_t,
            d_alpha_t,
            d_sigma_t,
        }
    }
}

pub trait Scheduler<B: Backend> {
    fn forward(&self, t: Tensor<B, 1>) -> SchedulerOutput<B>;
    fn snr_inverse(&self, snr: Tensor<B, 1>) -> Tensor<B, 1>;
}

#[derive(Module, Debug)]
pub struct CondOTScheduler<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> CondOTScheduler<B> {
    pub fn new() -> Self {
        Self {
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Scheduler<B> for CondOTScheduler<B> {
    fn forward(&self, t: Tensor<B, 1>) -> SchedulerOutput<B> {
        SchedulerOutput::new(
            t.clone(),
            Tensor::ones_like(&t) - t.clone(),
            Tensor::ones_like(&t),
            -Tensor::ones_like(&t),
        )
    }

    fn snr_inverse(&self, snr: Tensor<B, 1>) -> Tensor<B, 1> {
        snr
    }
}

#[derive(Module, Debug)]
pub struct CosineScheduler<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> CosineScheduler<B> {
    pub fn new() -> Self {
        Self {
            _b: std::marker::PhantomData,
        }
    }

    // arctan function for tensors using taylor series expansion
    // since Burn doesn't provide a direct atan operation
    fn arctan(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        // taylor series expansion of arctan(x)
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
}

impl<B: Backend> Scheduler<B> for CosineScheduler<B> {
    fn forward(&self, t: Tensor<B, 1>) -> SchedulerOutput<B> {
        let pi: f32 = std::f32::consts::PI;
        let half_pi = pi / 2.0;
        
        let half_pi_t = t.clone() * half_pi.clone();
        
        SchedulerOutput::new(
            half_pi_t.clone().sin(),
            half_pi_t.clone().cos(),
            Tensor::full(t.dims(), half_pi, &t.device()) * half_pi_t.clone().cos(),
            -Tensor::full(t.dims(), half_pi, &t.device()) * half_pi_t.sin(),
        )
    }

    fn snr_inverse(&self, snr: Tensor<B, 1>) -> Tensor<B, 1> {
        let pi: f64 = std::f32::consts::PI.into();
        self.arctan(snr).mul_scalar(2.0 / pi)
    }
}