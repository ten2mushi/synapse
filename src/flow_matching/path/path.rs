use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use super::path_sample::{PathSample, ContinuousPathSample, DiscretePathSample};

pub trait ProbPath<B: Backend> {
    /// Sample from the probability path:
    /// Given (X_0,X_1) ~ π(X_0,X_1)
    /// Returns X_0, X_1, X_t ~ p_t(X_t) and conditional target Y
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>, 
        t: Tensor<B, 1>
    ) -> Result<Box<dyn PathSample<B>>, ProbPathError>;

    fn assert_sample_shape(
        &self,
        x_0: &Tensor<B, 2>,
        x_1: &Tensor<B, 2>,
        t: &Tensor<B, 1>
    ) -> Result<(), ProbPathError> {
        if t.dims().len() != 1 {
            return Err(ProbPathError::InvalidTimeShape(t.dims().to_vec()));
        }

        let batch_size = x_0.dims()[0];
        if t.dims()[0] != batch_size || x_1.dims()[0] != batch_size {
            return Err(ProbPathError::BatchSizeMismatch {
                expected: batch_size,
                got: t.dims()[0],
            });
        }

        Ok(())
    }
}

/// continuous probability path that transforms distribution
/// p(X_0) into p(X_1) over t=0→1 with continuous trajectories
pub trait ContinuousProbPath<B: Backend>: ProbPath<B> {
    /// sample from the continuous probability path:
    /// given (X_0,X_1) ~ π(X_0,X_1)
    /// returns X_0, X_1, X_t ~ p_t(X_t) and conditional target Y
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>
    ) -> Result<ContinuousPathSample<B>, ProbPathError>;
}

/// discrete probability path that transforms distribution
/// p(X_0) into p(X_1) over t=0→1 with discrete jumps
pub trait DiscreteProbPath<B: Backend>: ProbPath<B> {
    /// sample from the discrete probability path:
    /// given (X_0,X_1) ~ π(X_0,X_1)
    /// returns X_0, X_1, X_t ~ p_t(X_t)
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>
    ) -> Result<DiscretePathSample<B>, ProbPathError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProbPathError {
    #[error("Time vector t must have shape [batch_size]. Got shape: {0:?}")]
    InvalidTimeShape(Vec<usize>),
    
    #[error("Time dimension must match batch size {expected}. Got {got}")]
    BatchSizeMismatch {
        expected: usize,
        got: usize,
    },
    
    #[error("Backend error: {0}")]
    BackendError(String),
}