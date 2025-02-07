// src/flow_matching/solver/ode_solver.rs
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use crate::flow_matching::utils::ModelWrapper;
use crate::flow_matching::solver::Solver;

#[derive(Debug, Clone)]
pub struct ODESolver<B: Backend, M: ModelWrapper<B>> {
    velocity_model: M,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend, M: ModelWrapper<B>> Module<B> for ODESolver<B, M> {
    type Record = M::Record;

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        self.velocity_model.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            velocity_model: self.velocity_model.fork(device),
            _backend: std::marker::PhantomData,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            velocity_model: self.velocity_model.to_device(device),
            _backend: std::marker::PhantomData,
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.velocity_model.visit(visitor);
    }

    fn map<MAP: ModuleMapper<B>>(self, mapper: &mut MAP) -> Self {
        Self {
            velocity_model: self.velocity_model.map(mapper),
            _backend: std::marker::PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            velocity_model: self.velocity_model.load_record(record),
            _backend: std::marker::PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        self.velocity_model.into_record()
    }
}

impl<B: Backend, M: ModelWrapper<B>> ODESolver<B, M> {
    pub fn new(velocity_model: M) -> Self {
        Self {
            velocity_model,
            _backend: std::marker::PhantomData,
        }
    }

    fn euler_step(
        &self,
        x: Tensor<B, 2>,
        t: Tensor<B, 1>, 
        step_size: f32,
    ) -> Tensor<B, 2> {
        let velocity = self.velocity_model.forward(x.clone(), t);
        x + (velocity * step_size)
    }

    /// Computes the log likelihood of a conditional flow between x_0 and x_1 at time t
    /// using the instantaneous change of variables formula.
    ///
    /// The likelihood is computed based on the velocity field approximation and the
    /// change in state variables over time.
    ///
    /// # Arguments
    /// * `x_0` - Initial state tensor with shape (batch_size, dim)  
    /// * `x_1` - Target state tensor with shape (batch_size, dim)
    /// * `t` - Time points tensor with shape (batch_size)
    /// * `step_size` - Optional integration step size (default: 0.01)
    /// * `method` - Integration method to use ("euler" or "rk4")
    ///
    /// # Returns
    /// * `Result<Tensor<B, 1>, String>` - Log likelihood for each sample in the batch
    ///
    pub fn compute_likelihood(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>,
        step_size: Option<f32>,
        method: &str,
    ) -> Result<Tensor<B, 1>, String> {
        let device = x_0.device();
        let batch_size = x_0.dims()[0];
        let step_size = step_size.unwrap_or(0.01);

        // Initialize cumulative log likelihood tensor
        let mut log_likelihood = Tensor::zeros([batch_size], &device);
        let mut current_state = x_0.clone();

        // Number of integration steps
        let n_steps = ((t.clone().into_scalar().elem::<f32>() - 0.0) / step_size).ceil() as usize;

        for step in 0..n_steps {
            let current_t = Tensor::from_data([step as f32 * step_size], &device);
            
            // Get velocity field at current state and time
            let velocity = self.velocity_model.forward(current_state.clone(), current_t.clone());

            // Compute next state based on integration method
            let next_state = match method {
                "euler" => self.euler_step(current_state.clone(), current_t, step_size),
                _ => return Err(format!("Unsupported integration method: {}", method)),
            };

            // Compute divergence of velocity field
            let div_v = self.compute_divergence(velocity.clone(), current_state.clone())?;

            // Update log likelihood using stepwise contribution
            let time_contribution = div_v * Tensor::from_data([step_size], &device);
            log_likelihood = log_likelihood - time_contribution;

            current_state = next_state;
        }

        // Add log probability of initial distribution (standard normal prior)
        let log_p_x0 = self.compute_standard_normal_logprob(&x_0)?;
        log_likelihood = log_likelihood + log_p_x0;

        Ok(log_likelihood)
    }

    /// computes the divergence of the velocity field at a given point using finite differences
    fn compute_divergence(
        &self,
        velocity: Tensor<B, 2>, 
        x: Tensor<B, 2>
    ) -> Result<Tensor<B, 1>, String> {
        let eps = 1e-6_f32;
        let batch_size = x.dims()[0];
        let dim = x.dims()[1];
        let device = x.device();

        let mut div = Tensor::zeros([batch_size], &device);

        for d in 0..dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();

            let plus_perturbation = Tensor::from_data([[eps]], &device);
            let minus_perturbation = Tensor::from_data([[-eps]], &device);

            x_plus = x_plus.slice_assign([0..batch_size, d..d+1], plus_perturbation);
            x_minus = x_minus.slice_assign([0..batch_size, d..d+1], minus_perturbation);

            let time_zeros = Tensor::zeros([batch_size], &device);
            let v_plus = self.velocity_model.forward(x_plus, time_zeros.clone());
            let v_minus = self.velocity_model.forward(x_minus, time_zeros);

            let partial = (v_plus - v_minus) / (2.0 * eps);
            let partial_d = partial.slice([0..batch_size, d..d+1]).squeeze_dims(&[1]);
            div = div + partial_d;
        }

        Ok(div)
    }

    /// computes log probability under standard normal distribution, returning a 1D tensor
    /// with shape [batch_size] containing the log probabilities for each sample
    fn compute_standard_normal_logprob(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let device = x.device();
        
        let norm_squared: Tensor<B, 1> = x.clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze_dims(&[1]);
        
        let neg_half = Tensor::from_data([-0.5_f32], &device);
        let log_2pi = Tensor::from_data([-0.5_f32 * (2.0 * std::f32::consts::PI).ln()], &device);
        
        let scaled_norm = (neg_half * norm_squared.clone())
            .squeeze_dims(&[1]);
            
        let log_prob = scaled_norm + log_2pi
            .repeat_dim(0, norm_squared.dims()[0]);
            
        Ok(log_prob)
    }
}

impl<B: Backend, M: ModelWrapper<B>> Solver<B, M> for ODESolver<B, M> {
    fn sample(
        &self,
        x_init: Tensor<B, 2>,
        step_size: Option<f32>,
        method: &str,
        time_grid: Tensor<B, 1>,
        return_intermediates: bool,
    ) -> Result<Tensor<B, 2>, String> {
        let device = x_init.device();
        let mut current_state = x_init;
        let mut intermediates = Vec::new();
    
        if return_intermediates {
            intermediates.push(current_state.clone());
        }
    
        let step_size = step_size.unwrap_or(0.01);
        
        let max_time = time_grid.clone().max().into_scalar().elem::<f32>();
        let n_steps = ((max_time - 0.0) / step_size).ceil() as usize;
    
        for step in 0..n_steps {
            let t = Tensor::from_data([step as f32 * step_size], &device);
            current_state = match method {
                "euler" => self.euler_step(current_state.clone(), t, step_size),
                _ => return Err(format!("Unsupported integration method: {}", method))
            };
    
            if return_intermediates {
                intermediates.push(current_state.clone());
            }
        }
    
        if return_intermediates {
            Ok(Tensor::stack(intermediates, 0))
        } else {
            Ok(current_state)
        }
    }
    // fn compute_likelihood(&self) -> {
    // to implement
    // }

}