use burn::tensor::{backend::Backend, Tensor};
use super::path_sample::{PathSample, ContinuousPathSample, DiscretePathSample};

pub trait ProbPath<B: Backend> {
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

pub trait ContinuousProbPath<B: Backend>: ProbPath<B> {
    fn sample(
        &self,
        x_0: Tensor<B, 2>,
        x_1: Tensor<B, 2>,
        t: Tensor<B, 1>
    ) -> Result<ContinuousPathSample<B>, ProbPathError>;
}

pub trait DiscreteProbPath<B: Backend>: ProbPath<B> {
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
