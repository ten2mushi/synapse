use rand::Rng;
use burn::prelude::Backend;
use crate::wiring::v1::base::{Wiring, WiringError};

#[derive(Debug, Clone)]
pub struct FullyConnected<B: Backend> {
    pub wiring: Wiring<B>,
    self_connections: bool,
}

impl<B: Backend> FullyConnected<B> {

    pub fn new(
        units: usize,
        output_dim: Option<usize>,
        self_connections: bool,
        erev_init_seed: u64,
        device: B::Device,
    ) -> Result<Self, WiringError> {

        let mut wiring = Wiring::new(units, device, erev_init_seed);
        
        if let Some(out_dim) = output_dim {
            if out_dim > units {
                return Err(WiringError::InvalidParameter);
            }
            wiring.set_output_dim(out_dim);
            wiring.organize_layers(out_dim, 0, 0)?;
        } else {
            wiring.set_output_dim(units);
            wiring.organize_layers(units, 0, 0)?;
        }

        Ok(Self {
            wiring,
            self_connections,
        })
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {

        self.wiring.build(input_dim)?;

        self.initialize_internal_connections()?;
        self.initialize_sensory_connections()?;

        
        Ok(())
    }

    fn initialize_internal_connections(&mut self) -> Result<(), WiringError> {
        for src in 0..self.wiring.units {
            for dest in 0..self.wiring.units {
                if src == dest && !self.self_connections {
                    continue;
                }
                let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                self.wiring.add_synapse(src, dest, polarity)?;
            }
        }
        
        Ok(())
    }

    fn initialize_sensory_connections(&mut self) -> Result<(), WiringError> {
        let input_dim = self.wiring.input_dim.ok_or(WiringError::NetworkNotBuilt)?;
        
        for src in 0..input_dim {
            for dest in 0..self.wiring.units {
                let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                self.wiring.add_sensory_synapse(src, dest, polarity)?;
            }
        }
        
        Ok(())
    }

}

impl<B: Backend> FullyConnected<B> {
    pub fn builder(units: usize, device: B::Device) -> FullyConnectedBuilder<B> {
        FullyConnectedBuilder::new(units, device)
    }
}

#[derive(Debug)]
pub struct FullyConnectedBuilder<B: Backend> {
    units: usize,
    output_dim: Option<usize>,
    self_connections: bool,
    erev_init_seed: u64,
    device: B::Device,
}

impl<B: Backend> FullyConnectedBuilder<B> {
    pub fn new(units: usize, device: B::Device) -> Self {
        Self {
            units,
            output_dim: None,
            self_connections: true,
            erev_init_seed: 1111,
            device,
        }
    }

    pub fn with_output_dim(mut self, output_dim: usize) -> Self {
        self.output_dim = Some(output_dim);
        self
    }

    pub fn with_self_connections(mut self, enabled: bool) -> Self {
        self.self_connections = enabled;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.erev_init_seed = seed;
        self
    }

    pub fn build(self) -> Result<FullyConnected<B>, WiringError> {
        FullyConnected::new(
            self.units,
            self.output_dim,
            self.self_connections,
            self.erev_init_seed,
            self.device,
        )
    }
}