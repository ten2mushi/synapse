pub mod discrete_solver;
pub mod ode_solver;
pub mod utils;

use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
use crate::flow_matching::utils::ModelWrapper;

pub trait Solver<B: Backend, M: ModelWrapper<B>>: Module<B> {
    /// sample from the solver given initial conditions
    fn sample(
        &self,
        x_init: Tensor<B, 2>,
        step_size: Option<f32>,
        method: &str,
        time_grid: Tensor<B, 1>,
        return_intermediates: bool,
    ) -> Result<Tensor<B, 2>, String>;
}
