use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig
    },
    tensor::{
        backend::Backend, Distribution, Tensor, activation
    },
    config::Config,
};

pub struct WeightInit;

impl WeightInit {
    pub fn xavier_uniform<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
        let fan_in = shape[0] as f64;
        let fan_out = shape[1] as f64;

        let bound = (6.0 / (fan_in + fan_out)).sqrt();
        
        Tensor::random(
            shape,
            Distribution::Uniform(-bound, bound),
            device,
        )
    }

    pub fn uniform<B: Backend>(
        shape: [usize; 2],
        low: f64,
        high: f64,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        Tensor::random(
            shape,
            Distribution::Uniform(low, high),
            device,
        )
    }
    
    /// TODO
    pub fn orthogonal<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
        let tensor = Tensor::random(
            shape,
            Distribution::Normal(0.0, 1.0),
            device,
        );
        tensor
    }
}

#[derive(Debug, Clone)]
pub struct LSTMState<B: Backend> {
    pub hidden: Tensor<B, 2>,
    pub cell: Tensor<B, 2>,
}

#[derive(Config)]
pub struct LSTMCellConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct LSTMCell<B: Backend> {
    input_map: Linear<B>,
    recurrent_map: Linear<B>,
    hidden_size: usize,
}

impl LSTMCellConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LSTMCell<B> {
        let gate_size = 4 * self.hidden_size;
        
        let input_map = LinearConfig::new(self.input_size, gate_size)
            .with_bias(true)
            .init(device);
        
        let recurrent_map = LinearConfig::new(self.hidden_size, gate_size)
            .with_bias(false)
            .init(device);

        LSTMCell {
            input_map,
            recurrent_map,
            hidden_size: self.hidden_size,
        }
    }
}

impl<B: Backend> LSTMCell<B> {
    pub fn forward(&self, inputs: Tensor<B, 2>, state: LSTMState<B>) -> (Tensor<B, 2>, LSTMState<B>) {
        let output_state = state.hidden;
        let cell_state = state.cell;

        let z = self.input_map.forward(inputs) + self.recurrent_map.forward(output_state);

        let gate_size = self.hidden_size;
        let i = z.clone().narrow(1, 0 * gate_size, gate_size);
        let ig = z.clone().narrow(1, 1 * gate_size, gate_size);
        let fg = z.clone().narrow(1, 2 * gate_size, gate_size);
        let og = z.narrow(1, 3 * gate_size, gate_size);

        let input_activation = activation::tanh(i);
        let input_gate = activation::sigmoid(ig);
        let forget_gate = activation::sigmoid(fg + 1.0);
        let output_gate = activation::sigmoid(og);

        let new_cell = cell_state * forget_gate + input_activation * input_gate;
        let new_hidden = activation::tanh(new_cell.clone()) * output_gate;

        (new_hidden.clone(), LSTMState {
            hidden: new_hidden,
            cell: new_cell,
        })
    }

    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> LSTMState<B> {
        LSTMState {
            hidden: Tensor::zeros([batch_size, self.hidden_size], device),
            cell: Tensor::zeros([batch_size, self.hidden_size], device),
        }
    }
}