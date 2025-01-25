use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use crate::flow_matching::path::path::{ProbPath, ProbPathError};
use crate::flow_matching::path::path_sample::{PathSample, ContinuousPathSample};
use crate::flow_matching::scheduler::scheduler::Scheduler;

#[derive(Debug)]
pub struct AffineProbPath<B: Backend, S: Scheduler<B>> {
    scheduler: S,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend, S: Scheduler<B>> AffineProbPath<B, S> {
    pub fn new(scheduler: S) -> Self {
        Self {
            scheduler,
            _backend: std::marker::PhantomData,
        }
    }

    fn expand_tensor_like(input: &Tensor<B, 1>, expand_to: &Tensor<B, 2>) -> Tensor<B, 2> {
        let target_dims = expand_to.dims();
        input.clone()
            .unsqueeze_dim(1)
            .repeat_dim(1, target_dims[1])
    }

    fn compute_ratio(numerator: &Tensor<B, 1>, denominator: &Tensor<B, 1>) -> Tensor<B, 1> {
        let num_data: Tensor<B, 1> = numerator.clone().squeeze_dims(&[0]);
        let den_data: Tensor<B, 1> = denominator.clone().squeeze_dims(&[0]);
        let ratio = num_data.into_scalar().elem::<f32>() / den_data.into_scalar().elem::<f32>();
        
        Tensor::from_data([ratio], &numerator.device())
    }

    fn negate_tensor(tensor: &Tensor<B, 1>) -> Tensor<B, 1> {
        tensor.clone() * (-1.0f32)
    }

    pub fn target_to_velocity(
        &self,
        x_1: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let a_t = Self::compute_ratio(&scheduler_output.d_sigma_t, &scheduler_output.sigma_t);

        // b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t
        let numerator = scheduler_output.d_alpha_t.clone() * scheduler_output.sigma_t.clone() - 
                       scheduler_output.d_sigma_t.clone() * scheduler_output.alpha_t.clone();
        let b_t = Self::compute_ratio(&numerator, &scheduler_output.sigma_t);

        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &x_1);

        a_t_expanded * x_t + b_t_expanded * x_1
    }

    pub fn epsilon_to_velocity(
        &self,
        epsilon: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let a_t = Self::compute_ratio(&scheduler_output.d_alpha_t, &scheduler_output.alpha_t);
        let numerator = scheduler_output.d_sigma_t.clone() * scheduler_output.alpha_t.clone() - 
                       scheduler_output.d_alpha_t.clone() * scheduler_output.sigma_t.clone();
        let b_t = Self::compute_ratio(&numerator, &scheduler_output.alpha_t);

        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &epsilon);

        a_t_expanded * x_t + b_t_expanded * epsilon
    }

    pub fn velocity_to_target(
        &self,
        velocity: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let denom = scheduler_output.d_alpha_t.clone() * scheduler_output.sigma_t.clone() - 
                   scheduler_output.d_sigma_t.clone() * scheduler_output.alpha_t.clone();
        
        let a_t = Self::compute_ratio(
            &Self::negate_tensor(&scheduler_output.d_sigma_t), 
            &denom
        );
        let b_t = Self::compute_ratio(&scheduler_output.sigma_t, &denom);

        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &velocity);

        a_t_expanded * x_t + b_t_expanded * velocity
    }

    pub fn epsilon_to_target(
        &self,
        epsilon: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let a_t = Tensor::from_data([1.0f32], &scheduler_output.alpha_t.device());
        let a_t = Self::compute_ratio(&a_t, &scheduler_output.alpha_t);
        
        let b_t = Self::compute_ratio(
            &Self::negate_tensor(&scheduler_output.sigma_t),
            &scheduler_output.alpha_t
        );
    
        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &epsilon);
    
        a_t_expanded * x_t + b_t_expanded * epsilon
    }

    pub fn velocity_to_epsilon(
        &self,
        velocity: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let denom = scheduler_output.d_sigma_t.clone() * scheduler_output.alpha_t.clone() - 
                   scheduler_output.d_alpha_t.clone() * scheduler_output.sigma_t.clone();
        
        let a_t = Self::compute_ratio(
            &Self::negate_tensor(&scheduler_output.d_alpha_t),
            &denom
        );
        let b_t = Self::compute_ratio(&scheduler_output.alpha_t, &denom);

        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &velocity);

        a_t_expanded * x_t + b_t_expanded * velocity
    }

    pub fn target_to_epsilon(
        &self,
        x_1: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let a_t = Tensor::from_data([1.0f32], &scheduler_output.sigma_t.device());
        let a_t = Self::compute_ratio(&a_t, &scheduler_output.sigma_t);
        
        let b_t = Self::compute_ratio(
            &Self::negate_tensor(&scheduler_output.alpha_t),
            &scheduler_output.sigma_t
        );
    
        let a_t_expanded = Self::expand_tensor_like(&a_t, &x_t);
        let b_t_expanded = Self::expand_tensor_like(&b_t, &x_1);
    
        a_t_expanded * x_t + b_t_expanded * x_1
    }
}


impl<B: Backend, S: Scheduler<B>> ProbPath<B> for AffineProbPath<B, S> {
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Result<Box<dyn PathSample<B>>, ProbPathError> {
        self.assert_sample_shape(&x_0, &x_1, &t)?;

        let scheduler_output = self.scheduler.forward(t.clone());
        
        let alpha_t = Self::expand_tensor_like(&scheduler_output.alpha_t, &x_1);
        let sigma_t = Self::expand_tensor_like(&scheduler_output.sigma_t, &x_1);
        let d_alpha_t = Self::expand_tensor_like(&scheduler_output.d_alpha_t, &x_1);
        let d_sigma_t = Self::expand_tensor_like(&scheduler_output.d_sigma_t, &x_1);

        let x_t = sigma_t * x_0.clone() + alpha_t * x_1.clone();
        let dx_t = d_sigma_t * x_0.clone() + d_alpha_t * x_1.clone();

        Ok(Box::new(ContinuousPathSample::new(x_1, x_0, t, x_t, dx_t)))
    }
}
