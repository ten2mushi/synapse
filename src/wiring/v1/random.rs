use rand::{
    rngs::StdRng, Rng, SeedableRng
};
use burn::prelude::Backend;
use crate::wiring::v1::base::{Wiring, WiringError};

#[derive(Debug, Clone)]
pub struct Random<B: Backend> {
    pub wiring: Wiring<B>,
    pub sparsity_level: f64,
    pub random_seed: u64,
}

impl<B: Backend> Random<B> {
    pub fn new(
        units: usize,
        output_dim: Option<usize>,
        sparsity_level: f64,
        random_seed: u64,
        device: B::Device,
    ) -> Result<Self, WiringError> {
        if sparsity_level < 0.0 || sparsity_level >= 1.0 {
            return Err(WiringError::InvalidParameter);
        }

        let output_dim = output_dim.unwrap_or(units);
        if output_dim > units {
            return Err(WiringError::InvalidParameter);
        }

        let mut wiring = Wiring::new(units, device, random_seed);
        
        wiring.set_output_dim(output_dim);
        
        let motor_neurons = output_dim;
        let remaining_neurons = units - motor_neurons;
        let command_neurons = remaining_neurons / 2;
        let inter_neurons = remaining_neurons - command_neurons;

        wiring.organize_layers(motor_neurons, command_neurons, inter_neurons)?;

        let mut random = Self {
            wiring,
            sparsity_level,
            random_seed,
        };

        random.initialize_internal_connections()?;

        Ok(random)
    }

    pub fn builder(units: usize, device: B::Device) -> RandomBuilder<B> {
        RandomBuilder::new(units, device)
    }

    fn initialize_internal_connections(&mut self) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        let total_possible_synapses = self.wiring.units * self.wiring.units;
        let number_of_synapses = (total_possible_synapses as f64 * (1.0 - self.sparsity_level))
            .round() as usize;

        let mut available_connections: Vec<(usize, usize)> = (0..self.wiring.units)
            .flat_map(|src| (0..self.wiring.units).map(move |dest| (src, dest)))
            .collect();

        for i in (1..available_connections.len()).rev() {
            let j = rng.gen_range(0..=i);
            available_connections.swap(i, j);
        }

        let selected_connections = &available_connections[0..number_of_synapses];

        for &(src, dest) in selected_connections {
            let polarity = if rng.gen_bool(0.80) { 1 } else { -1 };
            self.wiring.add_synapse(src, dest, polarity)?;
        }

        Ok(())
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        self.wiring.build(input_dim)?;

        let total_possible_sensory = input_dim * self.wiring.units;
        let number_of_sensory_synapses = 
            (total_possible_sensory as f64 * (1.0 - self.sparsity_level)).round() as usize;

        let mut available_sensory: Vec<(usize, usize)> = (0..input_dim)
            .flat_map(|src| (0..self.wiring.units).map(move |dest| (src, dest)))
            .collect();

        for i in (1..available_sensory.len()).rev() {
            let j = rng.gen_range(0..=i);
            available_sensory.swap(i, j);
        }

        let selected_sensory = &available_sensory[0..number_of_sensory_synapses];

        for &(src, dest) in selected_sensory {
            let polarity = if rng.gen_bool(0.80) { 1 } else { -1 };
            self.wiring.add_sensory_synapse(src, dest, polarity)?;
        }

        Ok(())
    }

    pub fn as_wiring(&self) -> &Wiring<B> {
        &self.wiring
    }

    pub fn as_wiring_mut(&mut self) -> &mut Wiring<B> {
        &mut self.wiring
    }

    pub fn sparsity_level(&self) -> f64 {
        self.sparsity_level
    }

    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }
}

#[derive(Debug)]
pub struct RandomBuilder<B: Backend> {
    units: usize,
    output_dim: Option<usize>,
    sparsity_level: f64,
    random_seed: u64,
    device: B::Device,
}

impl<B: Backend> RandomBuilder<B> {
    pub fn new(units: usize, device: B::Device) -> Self {
        Self {
            units,
            output_dim: None,
            sparsity_level: 0.0,
            random_seed: 1111,
            device,
        }
    }

    pub fn with_output_dim(mut self, output_dim: usize) -> Self {
        self.output_dim = Some(output_dim);
        self
    }

    pub fn with_sparsity_level(mut self, sparsity_level: f64) -> Self {
        self.sparsity_level = sparsity_level;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    pub fn build(self) -> Result<Random<B>, WiringError> {
        Random::new(
            self.units,
            self.output_dim,
            self.sparsity_level,
            self.random_seed,
            self.device,
        )
    }
}

impl<B: Backend> std::fmt::Display for Random<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Random Wiring (units: {}, sparsity: {:.2}, seed: {})",
            self.wiring.units,
            self.sparsity_level,
            self.random_seed
        )
    }
}