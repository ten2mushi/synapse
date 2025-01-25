use burn::tensor::{backend::Backend, ElementConversion, Tensor, Distribution, activation};

use crate::flow_matching::path::path::{ProbPath, ProbPathError};
use crate::flow_matching::path::path_sample::{PathSample, DiscretePathSample};
use crate::flow_matching::scheduler::scheduler::Scheduler;

#[derive(Debug)]
pub struct MixtureDiscreteProbPath<B: Backend, S: Scheduler<B>> {
    scheduler: S,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend, S: Scheduler<B>> MixtureDiscreteProbPath<B, S> {
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

    pub fn posterior_to_velocity(
        &self,
        posterior_logits: Tensor<B, 2>,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let scheduler_output = self.scheduler.forward(t);
        
        let kappa_t = scheduler_output.alpha_t;
        let d_kappa_t = scheduler_output.d_alpha_t;
        
        let posterior = activation::softmax(posterior_logits, 1);
        let vocab_size = posterior.dims()[1];
        
        let x_t_indices = x_t.clone().argmax(1);
        
        let x_t_one_hot = Tensor::<B, 2>::zeros([x_t.dims()[0], vocab_size], &x_t.device());
        
        for i in 0..x_t.dims()[0] {
            let idx = x_t_indices.clone().slice([i..i+1]).into_scalar().elem::<i64>() as usize;
            x_t_one_hot.clone().slice_assign([i..i+1, idx..idx+1], Tensor::from_data([1.0], &x_t.device()));
        }
        
        let coeff = d_kappa_t / (Tensor::from_data([1.0f32], &kappa_t.device()) - kappa_t);
        let coeff_expanded = Self::expand_tensor_like(&coeff, &posterior);
        
        coeff_expanded * (posterior - x_t_one_hot)
    }
}

impl<B: Backend, S: Scheduler<B>> ProbPath<B> for MixtureDiscreteProbPath<B, S> {
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>,
    ) -> Result<Box<dyn PathSample<B>>, ProbPathError> {
        self.assert_sample_shape(&x_0, &x_1, &t)?;
    
        let sigma_t = self.scheduler.forward(t.clone()).sigma_t;
        let sigma_t_expanded = Self::expand_tensor_like(&sigma_t, &x_1);
        
        let random_values = Tensor::random_like(&x_1, Distribution::Uniform(0.0, 1.0));
        
        let source_mask = random_values.lower_equal(sigma_t_expanded);
        let x_t = x_0.clone().mask_where(source_mask, x_1.clone());
    
        Ok(Box::new(DiscretePathSample::new(x_1, x_0, t, x_t)))
    }
}
