use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


pub trait PathSample<B: Backend> {
    /// target samples X_1
    fn x_1(&self) -> &Tensor<B, 2>;
    /// source samples X_0
    fn x_0(&self) -> &Tensor<B, 2>;
    /// time samples t
    fn t(&self) -> &Tensor<B, 1>;
    /// samples x_t
    fn x_t(&self) -> &Tensor<B, 2>;
}

#[derive(Module, Debug)]
pub struct ContinuousPathSample<B: Backend> {
    pub x_1: Tensor<B, 2>,
    pub x_0: Tensor<B, 2>,
    pub t: Tensor<B, 1>,
    pub x_t: Tensor<B, 2>,
    pub dx_t: Tensor<B, 2>,
}

impl<B: Backend> ContinuousPathSample<B> {
    pub fn new(
        x_1: Tensor<B, 2>,
        x_0: Tensor<B, 2>,
        t: Tensor<B, 1>,
        x_t: Tensor<B, 2>,
        dx_t: Tensor<B, 2>,
    ) -> Self {
        Self {
            x_1,
            x_0,
            t,
            x_t,
            dx_t,
        }
    }

    pub fn dx_t(&self) -> &Tensor<B, 2> {
        &self.dx_t
    }
}

#[derive(Module, Debug)]
pub struct DiscretePathSample<B: Backend> {
    pub x_1: Tensor<B, 2>,
    pub x_0: Tensor<B, 2>,
    pub t: Tensor<B, 1>,
    pub x_t: Tensor<B, 2>,
}

impl<B: Backend> DiscretePathSample<B> {
    pub fn new(
        x_1: Tensor<B, 2>,
        x_0: Tensor<B, 2>,
        t: Tensor<B, 1>,
        x_t: Tensor<B, 2>,
    ) -> Self {
        Self {
            x_1,
            x_0,
            t,
            x_t,
        }
    }
}

impl<B: Backend> PathSample<B> for ContinuousPathSample<B> {
    fn x_1(&self) -> &Tensor<B, 2> {
        &self.x_1
    }

    fn x_0(&self) -> &Tensor<B, 2> {
        &self.x_0
    }

    fn t(&self) -> &Tensor<B, 1> {
        &self.t
    }

    fn x_t(&self) -> &Tensor<B, 2> {
        &self.x_t
    }
}

impl<B: Backend> PathSample<B> for DiscretePathSample<B> {
    fn x_1(&self) -> &Tensor<B, 2> {
        &self.x_1
    }

    fn x_0(&self) -> &Tensor<B, 2> {
        &self.x_0
    }

    fn t(&self) -> &Tensor<B, 1> {
        &self.t
    }

    fn x_t(&self) -> &Tensor<B, 2> {
        &self.x_t
    }
}
