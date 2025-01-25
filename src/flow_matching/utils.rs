use burn::tensor::{backend::Backend, Tensor};
use burn::prelude::Module;
use crate::flow_matching::solver::Solver;
use crate::flow_matching::scheduler::scheduler::Scheduler;

pub trait ModelWrapper<B: Backend>: Module<B> {
    fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 1>) -> Tensor<B, 2>;
}

pub struct FlowMatch<B, M, S, O>
where
    B: Backend,
    M: ModelWrapper<B>,
    S: Scheduler<B>,
    O: Solver<B, M>,
{
    pub model: M,
    pub scheduler: S,
    pub solver: O,
    _backend: std::marker::PhantomData<B>,
}

impl<B, M, S, O> FlowMatch<B, M, S, O>
where
    B: Backend,
    M: ModelWrapper<B>,
    S: Scheduler<B>,
    O: Solver<B, M>,
{
    pub fn new(model: M, scheduler: S, solver: O) -> Self {
        Self {
            model,
            scheduler,
            solver,
            _backend: std::marker::PhantomData,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 1>) -> Tensor<B, 2> {
        self.model.forward(x, t)
    }
}

pub struct FlowMatchBuilder<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FlowMatchBuilder<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn build<M, S, O>(
        self,
        model: M,
        scheduler: S,
        solver: O,
    ) -> FlowMatch<B, M, S, O>
    where
        M: ModelWrapper<B>,
        S: Scheduler<B>, 
        O: Solver<B, M>,
    {
        FlowMatch::new(model, scheduler, solver)
    }
}