use rand::{
    rngs::StdRng, Rng, SeedableRng
};
use burn::prelude::Backend;
use crate::wiring::v1::base::{Wiring, WiringError};

#[derive(Debug, Clone)]
pub struct NCP<B: Backend> {
    pub wiring: Wiring<B>,
    pub num_inter_neurons: usize,
    pub num_command_neurons: usize,
    pub num_motor_neurons: usize,
    pub sensory_fanout: usize,
    pub inter_fanout: usize,
    pub recurrent_command_synapses: usize,
    pub motor_fanin: usize,
    // rng: StdRng,
    pub random_seed: u64,
    pub motor_neurons: Vec<usize>,
    pub command_neurons: Vec<usize>,
    pub inter_neurons: Vec<usize>,
    pub sensory_neurons: Option<Vec<usize>>,
}

impl<B: Backend> NCP<B> {

    // :param inter_neurons: The number of inter neurons (layer 2)
    // :param command_neurons: The number of command neurons (layer 3)
    // :param motor_neurons: The number of motor neurons (layer 4 = number of outputs)
    // :param sensory_fanout: The average number of outgoing synapses from the sensory to the inter neurons
    // :param inter_fanout: The average number of outgoing synapses from the inter to the command neurons
    // :param recurrent_command_synapses: The average number of recurrent connections in the command neuron layer
    // :param motor_fanin: The average number of incoming synapses of the motor neurons from the command neurons

    pub fn new(
        inter_neurons: usize,
        command_neurons: usize,
        motor_neurons: usize,
        sensory_fanout: usize,
        inter_fanout: usize,
        recurrent_command_synapses: usize,
        motor_fanin: usize,
        seed: u64,
        device: B::Device,
    ) -> Result<Self, WiringError> {

        let total_units = inter_neurons + command_neurons + motor_neurons;
        let mut wiring = Wiring::new(total_units, device, seed);

        wiring.set_output_dim(motor_neurons);
        
        wiring.organize_layers(motor_neurons, command_neurons, inter_neurons)?;

        if motor_fanin > command_neurons {
            return Err(WiringError::InvalidParameter);
        }
        if sensory_fanout > inter_neurons {
            return Err(WiringError::InvalidParameter);
        }
        if inter_fanout > command_neurons {
            return Err(WiringError::InvalidParameter);
        }

        let motor_neurons = wiring.motor_neurons.clone();
        let command_neurons = wiring.command_neurons.clone();
        let inter_neurons = wiring.inter_neurons.clone();

        Ok(Self {
            wiring,
            num_inter_neurons: inter_neurons.len(),
            num_command_neurons: command_neurons.len(),
            num_motor_neurons: motor_neurons.len(),
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            random_seed: seed,
            motor_neurons,
            command_neurons,
            inter_neurons,
            sensory_neurons: None,
        })
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {
        self.wiring.build(input_dim)?;
        self.sensory_neurons = Some((0..input_dim).collect());

        self.build_sensory_to_inter_layer()?;
        self.build_inter_to_command_layer()?;
        self.build_recurrent_command_layer()?;
        self.build_command_to_motor_layer()?;

        Ok(())
    }

    fn build_sensory_to_inter_layer(&mut self) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        let mut unreachable_inter_neurons = self.inter_neurons.clone();
        let sensory_neurons = self.sensory_neurons
            .as_ref()
            .ok_or(WiringError::InitializationError(
                "Sensory neurons not initialized".to_string()
            ))?;

        for &src in sensory_neurons.iter() {
            let dest_indices: Vec<usize> = (0..self.sensory_fanout)
                .map(|_| {
                    let idx = rng.gen_range(0..self.inter_neurons.len());
                    self.inter_neurons[idx]
                })
                .collect();

            for &dest in dest_indices.iter() {
                unreachable_inter_neurons.retain(|&x| x != dest);
                let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                self.wiring.add_sensory_synapse(src, dest, polarity)?;
            }
        }

        if !unreachable_inter_neurons.is_empty() {
            let mean_fanin = std::cmp::max(
                1,
                (sensory_neurons.len() * self.sensory_fanout) / self.num_inter_neurons
            );
            let fanin = std::cmp::min(mean_fanin, sensory_neurons.len());

            for &dest in unreachable_inter_neurons.iter() {
                for _ in 0..fanin {
                    let src_idx = rng.gen_range(0..sensory_neurons.len());
                    let src = sensory_neurons[src_idx];
                    let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                    self.wiring.add_sensory_synapse(src, dest, polarity)?;
                }
            }
        }

        Ok(())
    }

    fn build_inter_to_command_layer(&mut self) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        let mut unreachable_command_neurons = self.command_neurons.clone();

        for &src in self.inter_neurons.iter() {
            let dest_indices: Vec<usize> = (0..self.inter_fanout)
                .map(|_| {
                    let idx = rng.gen_range(0..self.command_neurons.len());
                    self.command_neurons[idx]
                })
                .collect();

            for &dest in dest_indices.iter() {
                unreachable_command_neurons.retain(|&x| x != dest);
                let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                self.wiring.add_synapse(src, dest, polarity)?;
            }
        }

        if !unreachable_command_neurons.is_empty() {
            let mean_fanin = std::cmp::max(
                1,
                (self.num_inter_neurons * self.inter_fanout) / self.num_command_neurons
            );
            let fanin = std::cmp::min(mean_fanin, self.num_command_neurons);

            for &dest in unreachable_command_neurons.iter() {
                for _ in 0..fanin {
                    let src_idx = rng.gen_range(0..self.inter_neurons.len());
                    let src = self.inter_neurons[src_idx];
                    let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                    self.wiring.add_synapse(src, dest, polarity)?;
                }
            }
        }

        Ok(())
    }

    fn build_recurrent_command_layer(&mut self) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        for _ in 0..self.recurrent_command_synapses {
            let src_idx = rng.gen_range(0..self.command_neurons.len());
            let dest_idx = rng.gen_range(0..self.command_neurons.len());
            let src = self.command_neurons[src_idx];
            let dest = self.command_neurons[dest_idx];
            let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
            self.wiring.add_synapse(src, dest, polarity)?;
        }
        Ok(())
    }

    fn build_command_to_motor_layer(&mut self) -> Result<(), WiringError> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        let mut unreachable_command_neurons = self.command_neurons.clone();

        for &dest in self.motor_neurons.iter() {
            let src_indices: Vec<usize> = (0..self.motor_fanin)
                .map(|_| {
                    let idx = rng.gen_range(0..self.command_neurons.len());
                    self.command_neurons[idx]
                })
                .collect();

            for &src in src_indices.iter() {
                unreachable_command_neurons.retain(|&x| x != src);
                let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                self.wiring.add_synapse(src, dest, polarity)?;
            }
        }

        if !unreachable_command_neurons.is_empty() {
            let mean_fanout = std::cmp::max(
                1,
                (self.num_motor_neurons * self.motor_fanin) / self.num_command_neurons
            );
            let fanout = std::cmp::min(mean_fanout, self.num_motor_neurons);

            for &src in unreachable_command_neurons.iter() {
                for _ in 0..fanout {
                    let dest_idx = rng.gen_range(0..self.motor_neurons.len());
                    let dest = self.motor_neurons[dest_idx];
                    let polarity = if self.wiring.rng.gen_bool(0.80) { 1 } else { -1 };
                    self.wiring.add_synapse(src, dest, polarity)?;
                }
            }
        }

        Ok(())
    }

    pub fn num_layers(&self) -> usize {
        3
    }

    pub fn get_neurons_of_layer(&self, layer_id: usize) -> Result<&[usize], String> {
        match layer_id {
            0 => Ok(&self.inter_neurons),
            1 => Ok(&self.command_neurons),
            2 => Ok(&self.motor_neurons),
            _ => Err(format!("Unknown layer {}", layer_id)),
        }
    }

    pub fn get_type_of_neuron(&self, neuron_id: usize) -> &str {
        if neuron_id < self.num_motor_neurons {
            "motor"
        } else if neuron_id < self.num_motor_neurons + self.num_command_neurons {
            "command"
        } else {
            "inter"
        }
    }
}

impl<B: Backend> AsRef<Wiring<B>> for NCP<B> {
    fn as_ref(&self) -> &Wiring<B> {
        &self.wiring
    }
}

impl<B: Backend> AsMut<Wiring<B>> for NCP<B> {
    fn as_mut(&mut self) -> &mut Wiring<B> {
        &mut self.wiring
    }
}