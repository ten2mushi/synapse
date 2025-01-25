use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig, 
        Dropout, DropoutConfig,
        Gelu, Tanh, Sigmoid,
    },
    tensor::{
        backend::Backend, Tensor
    },
};

#[derive(Module, Debug)]
struct BackboneLayer<B: Backend> {
    linear: Linear<B>,
    activation: Gelu,
    dropout: Dropout,
}

impl<B: Backend> BackboneLayer<B> {
    fn new(input_size: usize, output_size: usize, dropout_prob: f64, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(input_size, output_size).init(device),
            activation: Gelu::new(),
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        let x = self.activation.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Config)]
pub struct CfCCellConfig {
    input_size: usize,
    hidden_size: usize,
    #[config(default = 128)]
    backbone_units: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl CfCCellConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_size == 0 {
            return Err("input_size must be greater than 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct CfCCell<B: Backend> {
    backbone: Option<BackboneLayer<B>>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    time_a: Linear<B>,
    time_b: Linear<B>,
    tanh: Tanh,
    sigmoid: Sigmoid,
}

impl<B: Backend> CfCCell<B> {
    pub fn new(config: CfCCellConfig, device: &B::Device) -> Self {
        config.validate().expect("Invalid configuration");

        let cat_shape = config.input_size + config.hidden_size;
        let backbone_output = if config.backbone_units > 0 {
            config.backbone_units
        } else {
            cat_shape
        };

        let backbone = if config.backbone_units > 0 {
            Some(BackboneLayer::new(
                cat_shape,
                config.backbone_units,
                config.dropout,
                device,
            ))
        } else {
            None
        };

        let ff1 = LinearConfig::new(backbone_output, config.hidden_size).init(device);
        let ff2 = LinearConfig::new(backbone_output, config.hidden_size).init(device);
        let time_a = LinearConfig::new(backbone_output, config.hidden_size).init(device);
        let time_b = LinearConfig::new(backbone_output, config.hidden_size).init(device);

        Self {
            backbone,
            ff1,
            ff2,
            time_a,
            time_b,
            tanh: Tanh::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>, hx: Tensor<B, 2>, ts: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let tensors = vec![input, hx];
        let x = Tensor::cat(tensors, 1);
        
        let x = if let Some(ref backbone) = self.backbone {
            backbone.forward(x)
        } else {
            x
        };

        let ff1 = self.tanh.forward(self.ff1.forward(x.clone()));
        let ff2 = self.tanh.forward(self.ff2.forward(x.clone()));
        let t_a = self.time_a.forward(x.clone());
        let t_b = self.time_b.forward(x);
        let t_interp = self.sigmoid.forward(t_a * ts + t_b);
        
        let ones = Tensor::ones_like(&t_interp);
        let new_hidden = ff1 * (ones - t_interp.clone()) + t_interp * ff2;
        
        (new_hidden.clone(), new_hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    mod ndarray_tests {
        use super::*;
        use burn::backend::ndarray::NdArray;

        fn create_test_cell<B: Backend>(device: &B::Device) -> (CfCCell<B>, usize, usize) {
            let input_size = 4;
            let hidden_size = 6;
            
            let config = CfCCellConfig::new(input_size, hidden_size)
                .with_backbone_units(8)
                .with_dropout(0.0);
                
            let cell = CfCCell::new(config, device);
            (cell, input_size, hidden_size)
        }

        #[test]
        fn test_cfc_forward() {
            type B = NdArray;
            let device = Default::default();

            let (cell, input_size, hidden_size) = create_test_cell::<B>(&device);

            let batch_size = 2;
            let input = Tensor::<B, 2>::ones([batch_size, input_size], &device);
            let hidden = Tensor::<B, 2>::zeros([batch_size, hidden_size], &device);
            let ts = Tensor::<B, 2>::ones([batch_size, hidden_size], &device);

            let (output, new_hidden) = cell.forward(input, hidden, ts);

            assert_eq!(output.dims(), [batch_size, hidden_size]);
            assert_eq!(new_hidden.dims(), [batch_size, hidden_size]);
            
            let diff = (output - new_hidden).abs().max().into_scalar();
            assert!(diff < 1e-6);
        }

        #[test]
        fn test_backbone_variations() {
            type B = NdArray;
            let device = Default::default();

            let input_size = 4;
            let hidden_size = 6;
            let batch_size = 2;

            let config1 = CfCCellConfig::new(input_size, hidden_size);
            let cell1 = CfCCell::new(config1, &device);
            
            let config2 = CfCCellConfig::new(input_size, hidden_size)
                .with_backbone_units(8);
            let cell2 = CfCCell::new(config2, &device);

            let input = Tensor::<B, 2>::ones([batch_size, input_size], &device);
            let hidden = Tensor::<B, 2>::zeros([batch_size, hidden_size], &device);
            let ts = Tensor::<B, 2>::ones([batch_size, hidden_size], &device);

            for cell in [cell1, cell2] {
                let (output, _) = cell.forward(input.clone(), hidden.clone(), ts.clone());
                assert_eq!(output.dims(), [batch_size, hidden_size]);
            }
        }
    }

    #[test]
    fn test_config_validation() {
        let valid_config = CfCCellConfig::new(10, 20);
        assert!(valid_config.validate().is_ok());

        let invalid_config = CfCCellConfig::new(0, 20);
        assert!(invalid_config.validate().is_err());
    }
}