pub mod ltc_cell;

use burn::{
    module::Module,
    tensor::{
        backend::Backend, Tensor, ElementConversion
    },
    config::Config,
};

use crate::{
    wiring::v1::base::WiringImpl,
    cells::{
        v1::lstm::{
            LSTMCell, LSTMCellConfig, LSTMState
        },
        v1::ltc::ltc_cell::{
            LTCCell, LTCCellConfig
        }
    },
};

#[derive(Config)]
pub struct LTCConfig {
    pub input_size: usize,
    #[config(default = true)]
    pub return_sequences: bool,
    #[config(default = true)]
    pub batch_first: bool,
    #[config(default = false)]
    pub mixed_memory: bool,
    pub input_mapping: String,
    pub output_mapping: String,
    #[config(default = 6)]
    pub ode_unfolds: usize,  
    #[config(default = 1e-8)]
    pub epsilon: f32,
    #[config(default = true)]
    pub implicit_param_constraints: bool,
    pub integration_method: String,
    #[config(default = 1e-5)]
    pub tolerance: f32,
    #[config(default = 42)]
    pub seed: u64
}


#[derive(Debug, Clone)]
pub struct LTCState<B: Backend> {
    pub h_state: Tensor<B, 2>,
    pub c_state: Option<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct LTC<B: Backend> {
    rnn_cell: LTCCell<B>,
    lstm: Option<LSTMCell<B>>,
    state_size: usize,
    sensory_size: usize,
    motor_size: usize,
    return_sequences: bool,
    batch_first: bool,
    mixed_memory: bool,
}

impl<B: Backend> LTC<B> {
    pub fn new(
        config: LTCConfig,
        device: &B::Device,
        wiring: WiringImpl<B>,
    ) -> Self {

        let state_size = wiring.units();
        let sensory_size = wiring.input_dim()
            .expect("Wiring must have input dimension defined");
        let motor_size = wiring.output_dim()
            .expect("Wiring must have output dimension defined");

        let cell_config = LTCCellConfig {
            input_size: config.input_size,
            input_mapping: config.input_mapping,
            output_mapping: config.output_mapping,
            ode_unfolds: config.ode_unfolds,
            epsilon: config.epsilon,
            implicit_param_constraints: config.implicit_param_constraints,
            integration_method: config.integration_method,
            tolerance: config.tolerance,
        };

        let rnn_cell = cell_config.init(device, wiring, motor_size);

        let lstm = if config.mixed_memory {
            let lstm_config = LSTMCellConfig {
                input_size: config.input_size,
                hidden_size: state_size,
                dropout: 0.0,
            };
            Some(lstm_config.init::<B>(device))
        } else {
            None
        };

        Self {
            rnn_cell,
            lstm,
            state_size,
            sensory_size,
            motor_size,
            return_sequences: config.return_sequences,
            batch_first: config.batch_first,
            mixed_memory: config.mixed_memory,
        }
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn sensory_size(&self) -> usize {
        self.sensory_size
    }

    pub fn motor_size(&self) -> usize {
        self.motor_size
    }

    pub fn output_size(&self) -> usize {
        self.motor_size
    }

    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> LTCState<B> {
        let h_state = Tensor::zeros([batch_size, self.state_size], device);
        let c_state = if self.mixed_memory {
            Some(Tensor::zeros([batch_size, self.state_size], device))
        } else {
            None
        };
        LTCState { h_state, c_state }
    }

    // pub fn forward(
    //     &self,
    //     input: Tensor<B, 3>,
    //     state: Option<LTCState<B>>,
    //     timespans: Option<Tensor<B, 2>>,
    // ) -> (Tensor<B, 3>, LTCState<B>) {
    //     let device = input.device();

    //     // print input size:
    //     let input_size = input.dims()[2];
    //     println!("Input size: {}", input_size);
    //     println!("Input: {:?}", input.clone());
        
    //     let (batch_size, seq_len, _input_size) = if self.batch_first {
    //         let dims = input.dims();
    //         (dims[0], dims[1], dims[2])
    //     } else {
    //         let dims = input.dims();
    //         (dims[1], dims[0], dims[2])
    //     };

    //     let mut current_state = state.unwrap_or_else(|| self.init_state(batch_size, &device));
    //     let mut output_sequence = Vec::with_capacity(seq_len);

    //     for t in 0..seq_len {
    //         let timestep_input = if self.batch_first {
    //             input.clone().narrow(1, t, 1).squeeze_dims::<2>(&[1])
    //         } else {
    //             input.clone().narrow(0, t, 1).squeeze_dims::<2>(&[0])
    //         };

    //         let elapsed_time = timespans.as_ref()
    //         .map(|ts| {
    //             ts.clone()
    //               .narrow(1, t, 1)
    //               .mean()
    //               .mean()
    //               .into_scalar()
    //               .elem::<f32>()
    //         })
    //         .unwrap_or(1.0);

    //         if self.mixed_memory {
    //             if let Some(lstm) = &self.lstm {
    //                 let lstm_state = LSTMState {
    //                     hidden: current_state.h_state.clone(),
    //                     cell: current_state.c_state
    //                         .clone()
    //                         .expect("LSTM cell state should exist when mixed_memory is true"),
    //                 };
                    
    //                 let (new_hidden, new_state) = lstm.forward(timestep_input.clone(), lstm_state);
    //                 current_state.h_state = new_hidden;
    //                 current_state.c_state = Some(new_state.cell);
    //             }
    //         }

    //         let (h_out, h_state) = self.rnn_cell.forward(
    //             timestep_input.clone(),
    //             current_state.h_state.clone(),
    //             elapsed_time,
    //             Some(self.rnn_cell.tolerance),
    //         );

    //         current_state.h_state = h_state;

    //         if self.return_sequences {
    //             output_sequence.push(h_out);
    //         }
    //     }

    //     let output = if self.return_sequences {
    //         let stack_dim = if self.batch_first { 1 } else { 0 };
    //         Tensor::stack(output_sequence, stack_dim)
    //     } else {
    //         output_sequence.last()
    //             .unwrap()
    //             .clone()
    //             .unsqueeze_dim(if self.batch_first { 1 } else { 0 })
    //     };

    //     (output, current_state)
    // }
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<LTCState<B>>,
        timespans: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, LTCState<B>) {
        let device = input.device();
        
        let (batch_size, seq_len, _input_size) = if self.batch_first {
            let dims = input.dims();
            (dims[0], dims[1], dims[2])
        } else {
            let dims = input.dims();
            (dims[1], dims[0], dims[2])
        };
    
        let mut current_state = state.unwrap_or_else(|| self.init_state(batch_size, &device));
        let mut output_sequence = Vec::with_capacity(seq_len);
        let mut last_output = None;
    
        for t in 0..seq_len {
            let timestep_input = if self.batch_first {
                input.clone().narrow(1, t, 1).squeeze_dims::<2>(&[1])
            } else {
                input.clone().narrow(0, t, 1).squeeze_dims::<2>(&[0])
            };
    
            let elapsed_time = timespans.as_ref()
                .map(|ts| {
                    ts.clone()
                        .narrow(1, t, 1)
                        .mean()
                        .mean()
                        .into_scalar()
                        .elem::<f32>()
                })
                .unwrap_or(1.0);
    
            if self.mixed_memory {
                if let Some(lstm) = &self.lstm {
                    let lstm_state = LSTMState {
                        hidden: current_state.h_state.clone(),
                        cell: current_state.c_state
                            .clone()
                            .expect("LSTM cell state should exist when mixed_memory is true"),
                    };
                    
                    let (new_hidden, new_state) = lstm.forward(timestep_input.clone(), lstm_state);
                    current_state.h_state = new_hidden;
                    current_state.c_state = Some(new_state.cell);
                }
            }
    
            let (h_out, h_state) = self.rnn_cell.forward(
                timestep_input,
                current_state.h_state,
                elapsed_time,
                Some(self.rnn_cell.tolerance),
            );
    
            current_state.h_state = h_state;
            last_output = Some(h_out.clone());
    
            if self.return_sequences {
                output_sequence.push(h_out);
            }
        }
    
        let output = if self.return_sequences {
            let stack_dim = if self.batch_first { 1 } else { 0 };
            Tensor::stack(output_sequence, stack_dim)
        } else {
            // Use last_output which is guaranteed to exist after the loop
            last_output.unwrap()
                .unsqueeze_dim(if self.batch_first { 1 } else { 0 })
        };
    
        (output, current_state)
    }
}