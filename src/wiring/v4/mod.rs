// pub mod viz;
use burn::prelude::*;
use burn::tensor::backend::Backend;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ops::Range;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WiringError {
    #[error("Cannot add synapse from {0} when network has only {1} units")]
    InvalidSourceNeuron(usize, usize),
    #[error("Cannot add synapse to {0} when network has only {1} units")]
    InvalidDestinationNeuron(usize, usize),
    #[error("Cannot add synapse with polarity {0} (expected -1 or +1)")]
    InvalidPolarity(i32),
    #[error("Cannot add sensory synapse from {0} when input has only {1} features")]
    InvalidSensoryNeuron(usize, usize),
    #[error("Invalid Parameters")]
    InvalidParameter,
    #[error("Layer organization error: total neurons mismatch")]
    LayerOrganizationError,
    #[error("Duplicate layer name: {0}")]
    DuplicateLayerName(String),
    #[error("Layer not found: {0}")]
    LayerNotFound(String),
    #[error("Invalid sensory layer position - must be at start")]
    InvalidSensoryLayerPosition,
    #[error("Invalid motor layer position - must be at end")]
    InvalidMotorLayerPosition,
    #[error("{0}")]
    InitializationError(String),
}


#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum LayerType {
    Sensory,
    Hidden,
    Motor,
}

#[derive(Debug, Clone)]
pub enum Fanout {
    Forward(String, f64, f64),    // (target_layer, sparsity, polarity)
    Backward(String, f64, f64),   // (target_layer, sparsity, polarity)
    Recurrent(usize, f64),        // (count, polarity)
    LongRange(String, f64, f64),  // (target_layer, sparsity, polarity)
    None,
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub name: String,
    pub layer_type: LayerType,
    pub num_units: usize,
    pub connections: Vec<Fanout>,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub layer_type: LayerType,
    pub id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub config: LayerConfig,
}


#[derive(Debug)]
struct FanoutPlan {
    source_name: String,
    target_name: String,
    source_range: Range<usize>,
    target_range: Range<usize>,
    connection_type: FanoutType,
    sensory: bool,
}

#[derive(Debug)]
enum FanoutType {
    Forward { sparsity: f64, polarity: f64 },
    Backward { sparsity: f64, polarity: f64 },
    Recurrent { count: usize, polarity: f64 },
    LongRange { sparsity: f64, polarity: f64 },
}

#[derive(Debug, Clone)]
pub struct Wiring<B: Backend> {
    pub input_size: usize,
    pub output_size: usize,
    pub total_units: usize,
    pub adjacency_matrix: Tensor<B, 2>,
    pub sensory_adjacency_matrix: Tensor<B, 2>,
    pub layers: Vec<Layer>,
    pub rng: StdRng,
}

impl<B: Backend> Wiring<B> {
    pub fn new(
        layer_configs: Vec<LayerConfig>,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, WiringError> {
        let mut wiring = Self::init_wiring(layer_configs, seed, device)?;
        wiring.build_connections(device)?;
        println!("Wiring initialized");
        Ok(wiring)
    }

    fn init_wiring(
        layer_configs: Vec<LayerConfig>,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, WiringError> {

        let mut current_sensory_idx = 0;
        let mut current_hidden_idx = 0;
        let mut sensory_id = 0;
        let mut hidden_id = 0;
        let mut layers = Vec::new();

        for config in layer_configs.iter() {

            match config.layer_type {
                LayerType::Sensory => {
                    let start_idx = current_sensory_idx;
                    current_sensory_idx += config.num_units;
                    let end_idx = current_sensory_idx;
                    let layer = Layer {
                        name: config.name.clone(),
                        layer_type: config.layer_type.clone(),
                        id: sensory_id,
                        start_idx,
                        end_idx,
                        config: config.clone(),
                    };
                    sensory_id += 1;
                    layers.push(layer);
                },
                LayerType::Hidden | LayerType::Motor => {
                    let start_idx = current_hidden_idx;
                    current_hidden_idx += config.num_units;
                    let end_idx = current_hidden_idx;
                    let layer = Layer {
                        name: config.name.clone(),
                        layer_type: config.layer_type.clone(),
                        id: hidden_id,
                        start_idx,
                        end_idx,
                        config: config.clone(),
                    };
                    hidden_id += 1;
                    layers.push(layer);
                },
            }
        }


        let sensory_units: usize = layers.iter()
            .filter(|l| matches!(l.layer_type, LayerType::Sensory))
            .map(|l| l.config.num_units)
            .sum();

        let motor_units: usize = layers.iter()
            .filter(|l| matches!(l.layer_type, LayerType::Motor))
            .map(|l| l.config.num_units)
            .sum();
        
        let hidden_units: usize = layers.iter()
            .filter(|l| matches!(l.layer_type, LayerType::Hidden | LayerType::Motor))
            .map(|l| l.config.num_units)
            .sum();


        Ok(Self {
            input_size: sensory_units,
            output_size: motor_units,
            total_units: hidden_units,
            adjacency_matrix: Tensor::zeros([hidden_units, hidden_units], device),
            sensory_adjacency_matrix: Tensor::zeros([sensory_units, hidden_units], device),
            layers,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    fn build_connections(
        &mut self,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let plans = self.create_connection_plans()?;
        
        for plan in plans {
            match plan.connection_type {
                FanoutType::Recurrent { count, polarity } => {
                    self.connect_recurrent_range(
                        plan.source_range,
                        plan.sensory,
                        count,
                        polarity,
                        device
                    )?;
                },
                FanoutType::Forward { sparsity, polarity } |
                FanoutType::Backward { sparsity, polarity } |
                FanoutType::LongRange { sparsity, polarity } => {
                    self.connect_range(
                        plan.source_range,
                        plan.target_range,
                        plan.sensory,
                        sparsity,
                        polarity,
                        device
                    )?;
                }
            }
        }
        Ok(())
    }

    fn create_connection_plans(&self) -> Result<Vec<FanoutPlan>, WiringError> {
        let mut plans = Vec::new();

        for layer in &self.layers {
            for conn in &layer.config.connections {

                let source_range:Vec<usize> = (layer.start_idx..layer.end_idx).collect();
                
                match conn {
                    Fanout::Forward(target, _sparsity, _polarity) |
                    Fanout::Backward(target, _sparsity, _polarity) |
                    Fanout::LongRange(target, _sparsity, _polarity) => {

                        let mut target_range: Vec<usize> = (0..0).collect();
                        for l in &self.layers {
                            if l.name == *target {
                                target_range = (l.start_idx..l.end_idx).collect();
                            }
                        }

                        let connection_type = match conn {
                            Fanout::Forward(_, s, p) => FanoutType::Forward {
                                sparsity: *s,
                                polarity: *p
                            },
                            Fanout::Backward(_, s, p) => FanoutType::Backward {
                                sparsity: *s,
                                polarity: *p
                            },
                            Fanout::LongRange(_, s, p) => FanoutType::LongRange {
                                sparsity: *s,
                                polarity: *p
                            },
                            _ => unreachable!()
                        };

                        plans.push(FanoutPlan {
                            source_name: layer.name.clone(),
                            target_name: target.clone(),
                            source_range: source_range[0]..source_range[source_range.len()-1]+1,
                            target_range: target_range[0]..target_range[target_range.len()-1]+1,
                            connection_type,
                            sensory: matches!(layer.layer_type, LayerType::Sensory),
                        });
                    },
                    Fanout::Recurrent(count, polarity) => {
                        plans.push(FanoutPlan {
                            source_name: layer.name.clone(),
                            target_name: layer.name.clone(),
                            source_range: source_range[0]..source_range[source_range.len()-1]+1,
                            target_range: source_range[0]..source_range[source_range.len()-1]+1,
                            connection_type: FanoutType::Recurrent {
                                count: *count,
                                polarity: *polarity
                            },
                            sensory: matches!(layer.layer_type, LayerType::Sensory),
                        });
                    },
                    Fanout::None => {}
                }
            }
        }
        Ok(plans)
    }

    fn connect_range(
        &mut self,
        source_range: Range<usize>,
        target_range: Range<usize>,
        sensory: bool,
        sparsity: f64,
        polarity: f64,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let num_source_neurons = source_range.end - source_range.start;
        let num_target_neurons = target_range.end - target_range.start;
        let fanout = std::cmp::max(
            (num_target_neurons as f64 * (1.0 - sparsity)) as usize,
            1
        );

        let mut unreachable_neurons: Vec<usize> = (target_range.start..target_range.end).collect();

        for src_offset in 0..num_source_neurons {
            let src = source_range.start + src_offset;
            let dest_indices: Vec<usize> = (0..fanout)
                .map(|_| {
                    let offset = self.rng.gen_range(0..num_target_neurons);
                    let dest = target_range.start + offset;
                    unreachable_neurons.retain(|&x| x != dest);
                    dest
                })
                .collect();

            for &dest in &dest_indices {
                let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
                if sensory {
                    self.add_sensory_synapse(src, dest, polarity_value, device)?;
                } else {
                    self.add_synapse(src, dest, polarity_value, device)?;
                }
            }
        }

        if !unreachable_neurons.is_empty() {
            let mean_fanin = std::cmp::max(
                (num_source_neurons * fanout) / num_target_neurons,
                1
            );
            
            for &dest in &unreachable_neurons {
                for _ in 0..mean_fanin {
                    let src_offset = self.rng.gen_range(0..num_source_neurons);
                    let src = source_range.start + src_offset;
                    let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
                    if sensory {
                        self.add_sensory_synapse(src, dest, polarity_value, device)?;
                    } else {
                        self.add_synapse(src, dest, polarity_value, device)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn connect_recurrent_range(
        &mut self,
        range: Range<usize>,
        sensory: bool,
        count: usize,
        polarity: f64,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let num_neurons = range.end - range.start;
        
        for _ in 0..count {
            let src_offset = self.rng.gen_range(0..num_neurons);
            let dest_offset = self.rng.gen_range(0..num_neurons);
            
            let src = range.start + src_offset;
            let dest = range.start + dest_offset;
            
            let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
            
            if sensory {
                self.add_sensory_synapse(src, dest, polarity_value, device)?;
            } else {
                self.add_synapse(src, dest, polarity_value, device)?;
            }
        }
        Ok(())
    }

    fn add_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        if src >= self.total_units {
            return Err(WiringError::InvalidSourceNeuron(src, self.total_units));
        }
        if dest >= self.total_units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.total_units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        let value = Tensor::from_data([[polarity as f32]], device);
        self.adjacency_matrix = self.adjacency_matrix.clone()
            .slice_assign([src..src + 1, dest..dest + 1], value);

        Ok(())
    }

    fn add_sensory_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let sensory_size = self.sensory_adjacency_matrix.dims()[0];
        
        if src >= sensory_size {
            return Err(WiringError::InvalidSensoryNeuron(src, sensory_size));
        }
        if dest >= self.total_units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.total_units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        let value = Tensor::from_data([[polarity as f32]], device);
        self.sensory_adjacency_matrix = self.sensory_adjacency_matrix.clone()
            .slice_assign([src..src + 1, dest..dest + 1], value);

        Ok(())
    }

    // Public interface methods
    // pub fn get_layer_neurons(&self, layer_name: &str) -> Option<&Vec<usize>> {
    //     self.neuron_map.get(layer_name)
    // }

    // pub fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
    //     self.layer_map.get(layer_name).copied()
    // }

    pub fn erev_initializer(&self) -> Tensor<B, 2> {
        self.adjacency_matrix.clone()
    }

    pub fn sensory_erev_initializer(&self) -> Tensor<B, 2> {
        self.sensory_adjacency_matrix.clone()
    }

    pub fn adjacency_matrix(&self) -> Tensor<B, 2> {
        self.adjacency_matrix.clone()
    }

    pub fn sensory_adjacency_matrix(&self) -> Tensor<B, 2> {
        self.sensory_adjacency_matrix.clone()
    }
}