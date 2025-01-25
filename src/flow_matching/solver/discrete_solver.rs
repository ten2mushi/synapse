// use burn::{
//     module::Module,
//     tensor::{backend::Backend, ElementConversion, Tensor},
// };
// use crate::flow_matching::{
//     path::mixture::MixtureDiscreteProbPath,
//     utils::ModelWrapper,
//     solver::Solver,
// };

// #[derive(Debug, Clone)]
// pub struct MixtureDiscreteEulerSolver<B: Backend, M: ModelWrapper<B>> {
//     model: M,
//     path: MixtureDiscreteProbPath<B>,
//     vocabulary_size: usize,
//     source_distribution_p: Option<Tensor<B, 1>>,
//     _backend: std::marker::PhantomData<B>,
// }

// impl<B: Backend, M: ModelWrapper<B>> Module<B> for MixtureDiscreteEulerSolver<B, M> {
//     type Record = M::Record;

//     fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
//         self.model.collect_devices(devices)
//     }

//     fn fork(self, device: &B::Device) -> Self {
//         Self {
//             model: self.model.fork(device),
//             path: self.path.fork(device),
//             vocabulary_size: self.vocabulary_size,
//             source_distribution_p: self.source_distribution_p.map(|p| p.to_device(device)),
//             _backend: std::marker::PhantomData,
//         }
//     }

//     fn to_device(self, device: &B::Device) -> Self {
//         self.fork(device)
//     }

//     fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
//         self.model.visit(visitor);
//         self.path.visit(visitor);
//     }

//     fn map<MAP: ModuleMapper<B>>(self, mapper: &mut MAP) -> Self {
//         Self {
//             model: self.model.map(mapper),
//             path: self.path.map(mapper),
//             vocabulary_size: self.vocabulary_size,
//             source_distribution_p: self.source_distribution_p,
//             _backend: std::marker::PhantomData,
//         }
//     }

//     fn load_record(self, record: Self::Record) -> Self {
//         Self {
//             model: self.model.load_record(record),
//             path: self.path,
//             vocabulary_size: self.vocabulary_size,
//             source_distribution_p: self.source_distribution_p,
//             _backend: std::marker::PhantomData,
//         }
//     }

//     fn into_record(self) -> Self::Record {
//         self.model.into_record()
//     }
// }