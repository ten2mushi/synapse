use burn::tensor::backend::Backend;
use crate::wiring::v1::{ncp::NCP, base::WiringError};

#[derive(Debug, Clone)]
pub struct AutoNCP<B: Backend> {
    pub ncp: NCP<B>,
    pub units: usize,
    pub output_size: usize,
    pub sparsity_level: f64,
    pub seed: u64,
}

impl<B: Backend> AutoNCP<B> {
    
    /// # Arguments
    /// * `units` - Total number of neurons in the network
    /// * `output_size` - Number of motor neurons (outputs)
    /// * `device` - Device for tensor operations
    /// * `sparsity_level` - Controls connection density (0.1 to 0.9), default 0.5
    /// * `seed` - Random seed for reproducibility
    /// 
    /// # Errors
    /// * Returns error if output_size is too large relative to units
    /// * Returns error if sparsity_level is outside valid range
    pub fn new(
        units: usize,
        output_size: usize,
        device: B::Device,
        sparsity_level: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, WiringError> {

        if output_size >= units - 2 {
            return Err(WiringError::InvalidParameter);
        }

        let sparsity_level = sparsity_level.unwrap_or(0.5);
        if sparsity_level < 0.0 || sparsity_level >= 1.0 {
            return Err(WiringError::InvalidParameter);
        }

        let seed = seed.unwrap_or(22222);
        
        let inter_and_command_neurons = units - output_size;
        let command_neurons = std::cmp::max(
            (0.4 * inter_and_command_neurons as f64) as usize,
            1
        );
        let inter_neurons = inter_and_command_neurons - command_neurons;

        let density_level = 1.0 - sparsity_level;
        let sensory_fanout = std::cmp::max(
            (inter_neurons as f64 * density_level) as usize,
            1
        );
        let inter_fanout = std::cmp::max(
            (command_neurons as f64 * density_level) as usize,
            1
        );
        let recurrent_command_synapses = std::cmp::max(
            (command_neurons as f64 * density_level * 2.0) as usize,
            1
        );
        let motor_fanin = std::cmp::max(
            (command_neurons as f64 * density_level) as usize,
            1
        );

        let ncp = NCP::new(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed,
            device,
        )?;

        Ok(Self {
            ncp,
            units,
            output_size,
            sparsity_level,
            seed,
        })
    }

    pub fn builder(units: usize, output_size: usize, device: B::Device) -> AutoNCPBuilder<B> {
        AutoNCPBuilder::new(units, output_size, device)
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {
        self.ncp.build(input_dim)
    }

    pub fn as_ncp(&self) -> &NCP<B> {
        &self.ncp
    }

    pub fn as_ncp_mut(&mut self) -> &mut NCP<B> {
        &mut self.ncp
    }

    pub fn units(&self) -> usize {
        self.units
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn sparsity_level(&self) -> f64 {
        self.sparsity_level
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Debug)]
pub struct AutoNCPBuilder<B: Backend> {
    units: usize,
    output_size: usize,
    device: B::Device,
    sparsity_level: Option<f64>,
    seed: Option<u64>,
}

impl<B: Backend> AutoNCPBuilder<B> {
    pub fn new(units: usize, output_size: usize, device: B::Device) -> Self {
        Self {
            units,
            output_size,
            device,
            sparsity_level: None,
            seed: None,
        }
    }

    pub fn with_sparsity_level(mut self, sparsity_level: f64) -> Self {
        self.sparsity_level = Some(sparsity_level);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> Result<AutoNCP<B>, WiringError> {
        AutoNCP::new(
            self.units,
            self.output_size,
            self.device,
            self.sparsity_level,
            self.seed,
        )
    }
}