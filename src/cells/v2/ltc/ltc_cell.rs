use burn::{
    config::Config,
    module::{
        Module, Param
    },
    tensor::{
        backend::Backend,
        activation, Distribution, Tensor,
    },
};

use crate::wiring::v2::Wiring;

const INIT_RANGES: [(f32, f32); 12] = [
    (0.001, 1.0),  // gleak
    (-0.2, 0.2),   // vleak
    (0.4, 0.6),    // cm
    (0.001, 1.0),  // w
    (3.0, 8.0),    // sigma
    (0.3, 0.8),    // mu
    (0.001, 1.0),  // sensory_w
    (3.0, 8.0),    // sensory_sigma
    (0.3, 0.8),    // sensory_mu
    (1.0, 10.0),    // tau_n
    (0.1, 1.0),     // n_inf_slope
    (-0.5, 0.5),    // n_inf_thresh
];

#[derive(Module, Debug, Clone)]
pub enum MappingType {
    None,
    Linear,
    Affine,
}

#[derive(Module, Debug, Clone)]
pub enum IntegrationMethod {
    Euler,
    // CashKarp,
    // BackwardEuler,
}

#[derive(Config)]
pub struct LTCCellConfig {
    pub input_size: usize,
    pub input_mapping: String,
    pub output_mapping: String,
    #[config(default = 6)]
    pub ode_unfolds: usize,
    #[config(default = 1e-8)]
    pub epsilon: f32,
    #[config(default = false)]
    pub implicit_param_constraints: bool,
    pub integration_method: String,
    #[config(default = 1e-5)]
    pub tolerance: f32,
}

pub struct LTCCellBuilder {
    config: LTCCellConfig,
}

impl LTCCellBuilder {
    pub fn new(input_size: usize) -> Self {
        Self {
            config: LTCCellConfig {
                input_size,
                input_mapping: "affine".to_string(),
                output_mapping: "affine".to_string(),
                ode_unfolds: 6,
                epsilon: 1e-8,
                implicit_param_constraints: false,
                integration_method: "euler".to_string(),
                tolerance: 1e-5,
            }
        }
    }

    pub fn with_input_mapping(mut self, mapping: &str) -> Self {
        self.config.input_mapping = mapping.to_string();
        self
    }

    pub fn with_output_mapping(mut self, mapping: &str) -> Self {
        self.config.output_mapping = mapping.to_string();
        self
    }

    pub fn with_ode_unfolds(mut self, unfolds: usize) -> Self {
        self.config.ode_unfolds = unfolds;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn with_implicit_constraints(mut self, implicit: bool) -> Self {
        self.config.implicit_param_constraints = implicit;
        self
    }

    pub fn with_integration_method(mut self, method: &str) -> Self {
        self.config.integration_method = method.to_string();
        self
    }

    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.config.tolerance = tol;
        self
    }

    pub fn build<B: Backend>(
        self,
        device: &B::Device,
        wiring: Wiring<B>,
        motor_size: usize,
    ) -> LTCCell<B> {
        self.config.init(device, wiring, motor_size)
    }
}


#[derive(Module, Debug)]
pub struct LTCCell<B: Backend> {
    gleak: Param<Tensor<B, 1>>,
    vleak: Param<Tensor<B, 1>>,
    cm: Param<Tensor<B, 1>>,

    w: Param<Tensor<B, 2>>,
    sigma: Param<Tensor<B, 2>>,
    mu: Param<Tensor<B, 2>>,

    sensory_w: Param<Tensor<B, 2>>,
    sensory_sigma: Param<Tensor<B, 2>>,
    sensory_mu: Param<Tensor<B, 2>>,

    erev: Param<Tensor<B, 2>>,
    sensory_erev: Param<Tensor<B, 2>>,

    tau_n: Param<Tensor<B, 1>>,
    n_inf_slope: Param<Tensor<B, 1>>,
    n_inf_thresh: Param<Tensor<B, 1>>,

    input_w: Option<Param<Tensor<B, 1>>>,
    input_b: Option<Param<Tensor<B, 1>>>,
    output_w: Option<Param<Tensor<B, 1>>>,
    output_b: Option<Param<Tensor<B, 1>>>,

    sparsity_mask: Option<Tensor<B, 2>>,
    sensory_sparsity_mask: Option<Tensor<B, 2>>,

    input_mapping: MappingType,
    output_mapping: MappingType,
    ode_unfolds: usize,
    epsilon: f32,
    implicit_param_constraints: bool,
    integration_method: IntegrationMethod,
    pub tolerance: f32,

    input_size: usize,
    state_size: usize,
    motor_size: usize,
}

impl LTCCellConfig {
    fn init_param<B: Backend, const D: usize>(
        param_idx: usize,
        shape: [usize; D],
        device: &B::Device,
    ) -> Tensor<B, D> {
        let (min, max) = INIT_RANGES.get(param_idx)
            .copied()
            .unwrap_or((0.0, 1.0));

        if (min - max).abs() < f32::EPSILON {
            Tensor::full(shape, min, device)
        } else {
            Tensor::random(shape, Distribution::Uniform(min.into(), max.into()), device)
        }
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        wiring: Wiring<B>,
        motor_size: usize,
    ) -> LTCCell<B> {
        let state_size = wiring.total_units;
        let input_size = self.input_size;

        // Parse mapping types
        let input_mapping = match self.input_mapping.as_str() {
            "linear" => MappingType::Linear,
            "affine" => MappingType::Affine,
            _ => MappingType::None,
        };

        let output_mapping = match self.output_mapping.as_str() {
            "linear" => MappingType::Linear,
            "affine" => MappingType::Affine,
            _ => MappingType::None,
        };

        let integration_method = match self.integration_method.as_str() {
            "euler" => IntegrationMethod::Euler,
            // "cash_karp" => IntegrationMethod::CashKarp,
            // "backward_euler" => IntegrationMethod::BackwardEuler,
            _ => IntegrationMethod::Euler,
        };

        let (input_w, input_b) = match input_mapping {
            MappingType::Linear | MappingType::Affine => {
                let w = Some(Param::from_tensor(Self::init_param(0, [input_size], device)));
                let b = match input_mapping {
                    MappingType::Affine => Some(Param::from_tensor(Tensor::zeros([input_size], device))),
                    _ => None,
                };
                (w, b)
            },
            MappingType::None => (None, None),
        };

        let (output_w, output_b) = match output_mapping {
            MappingType::Linear | MappingType::Affine => {
                let w = Some(Param::from_tensor(Self::init_param(1, [motor_size], device)));
                let b = match output_mapping {
                    MappingType::Affine => Some(Param::from_tensor(Tensor::zeros([motor_size], device))),
                    _ => None,
                };
                (w, b)
            },
            MappingType::None => (None, None),
        };

        LTCCell {
            gleak: Param::from_tensor(Self::init_param(0, [state_size], device)),
            vleak: Param::from_tensor(Self::init_param(1, [state_size], device)),
            cm: Param::from_tensor(Self::init_param(2, [state_size], device)),
            
            w: Param::from_tensor(Self::init_param(3, [state_size, state_size], device)),
            sigma: Param::from_tensor(Self::init_param(4, [state_size, state_size], device)),
            mu: Param::from_tensor(Self::init_param(5, [state_size, state_size], device)),
            
            sensory_w: Param::from_tensor(Self::init_param(6, [input_size, state_size], device)),
            sensory_sigma: Param::from_tensor(Self::init_param(7, [input_size, state_size], device)),
            sensory_mu: Param::from_tensor(Self::init_param(8, [input_size, state_size], device)),

            erev: Param::from_tensor(wiring.erev_initializer()),
            sensory_erev: Param::from_tensor(wiring.sensory_erev_initializer()),

            tau_n: Param::from_tensor(Self::init_param(9, [state_size], device)),
            n_inf_slope: Param::from_tensor(Self::init_param(10, [state_size], device)),
            n_inf_thresh: Param::from_tensor(Self::init_param(11, [state_size], device)),

            input_w,
            input_b,
            output_w,
            output_b,

            sparsity_mask: Some(wiring.adjacency_matrix()),
            sensory_sparsity_mask: Some(wiring.sensory_adjacency_matrix()),

            input_mapping,
            output_mapping,
            ode_unfolds: self.ode_unfolds,
            epsilon: self.epsilon,
            implicit_param_constraints: self.implicit_param_constraints,
            integration_method,
            tolerance: self.tolerance,

            input_size,
            state_size,
            motor_size,
        }
    }
}

impl<B: Backend> LTCCell<B> {
    fn _create_scalar_tensor(&self, value: f32, batch_size: usize, device: &B::Device) -> Tensor<B, 2> {
        Tensor::full([batch_size, 1], value, device)
    }

    fn make_positive_1d(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        if self.implicit_param_constraints {
            activation::softplus(x, 1.0)
        } else {
            x
        }
    }

    fn make_positive_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.implicit_param_constraints {
            activation::softplus(x, 1.0)
        } else {
            x
        }
    }

    // fn sigmoid(
    //     &self,
    //     v_pre: Tensor<B, 2>,
    //     mu: Tensor<B, 2>,
    //     sigma: Tensor<B, 2>,
    // ) -> Tensor<B, 3> {
    //     let v_pre = v_pre.unsqueeze_dim(2);
    //     let mu = mu.unsqueeze_dim(0);
    //     let sigma = sigma.unsqueeze_dim(0);
    //     let mues = v_pre - mu;
    //     let x = sigma * mues;
    //     activation::sigmoid(x)
    // }
    fn sigmoid(
        &self,
        v_pre: Tensor<B, 2>, 
        mu: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
     ) -> Tensor<B, 3> {
        activation::sigmoid(
            sigma.unsqueeze_dim(0) * 
            (v_pre.unsqueeze_dim(2) - mu.unsqueeze_dim(0))
        )
     }
     
    fn map_inputs(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = inputs;
        match self.input_mapping {
            MappingType::Linear | MappingType::Affine => {
                if let Some(w) = &self.input_w {
                    x = x * w.val().clone().unsqueeze_dim(0);
                }
                if let Some(b) = &self.input_b {
                    if matches!(self.input_mapping, MappingType::Affine) {
                        x = x + b.val().clone().unsqueeze_dim(0);
                    }
                }
            },
            MappingType::None => {}
        }
        x
    }

    fn map_outputs(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = if self.motor_size < self.state_size {
            state.slice([None, Some((0, self.motor_size as i64))])
        } else {
            state
        };

        match self.output_mapping {
            MappingType::Linear | MappingType::Affine => {
                if let Some(w) = &self.output_w {
                    x = x * w.val().clone().unsqueeze_dim(0);
                }
                if let Some(b) = &self.output_b {
                    if matches!(self.output_mapping, MappingType::Affine) {
                        x = x + b.val().clone().unsqueeze_dim(0);
                    }
                }
            },
            MappingType::None => {}
        }
        x
    }

    // V1
    // fn compute_sensory_effects(
    //     &self,
    //     inputs: &Tensor<B, 2>,
    // ) -> (Tensor<B, 2>, Tensor<B, 2>) {
    //     let sensory_w = self.make_positive_2d(self.sensory_w.val().clone());
    //     let sensory_activation = self.sigmoid(
    //         inputs.clone(),
    //         self.sensory_mu.val().clone(),
    //         self.sensory_sigma.val().clone(),
    //     );

    //     let mut sensory_w_activation = sensory_activation.clone()
    //         * sensory_w.unsqueeze_dim(0);

    //     if let Some(mask) = &self.sensory_sparsity_mask {
    //         sensory_w_activation = sensory_w_activation
    //             * mask.clone().unsqueeze_dim(0);
    //     }

    //     let sensory_rev_activation = sensory_w_activation.clone()
    //         * self.sensory_erev.val().clone().unsqueeze_dim(0);

    //     let w_numerator_sensory = sensory_rev_activation
    //         .sum_dim(1)
    //         .squeeze_dims(&[1]);
    //     let w_denominator_sensory = sensory_w_activation
    //         .sum_dim(1)
    //         .squeeze_dims(&[1]);

    //     (w_numerator_sensory, w_denominator_sensory)
    // }

    // BRRRRR
    fn compute_sensory_effects(
        &self,
        inputs: &Tensor<B, 2>,
     ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Combined weight and activation computation 
        let w_activation = {
            let w = self.make_positive_2d(self.sensory_w.val().clone());
            let act = self.sigmoid(
                inputs.clone(),
                self.sensory_mu.val().clone(), 
                self.sensory_sigma.val().clone()
            );
            let mut wa = act * w.unsqueeze_dim(0);
            
            // Apply mask if exists
            if let Some(mask) = &self.sensory_sparsity_mask {
                wa = wa * mask.clone().unsqueeze_dim(0);
            }
            wa
        };
     
        // Parallel numerator/denominator computation
        let num = (w_activation.clone() * self.sensory_erev.val().clone().unsqueeze_dim(0))
            .sum_dim(1)
            .squeeze_dims(&[1]);
        let den = w_activation
            .sum_dim(1)
            .squeeze_dims(&[1]);
     
        (num, den)
     }

    // V1
    // fn compute_states(
    //     &self,
    //     v_pre: Tensor<B, 2>,
    //     n_pre: Tensor<B, 2>,
    //     w_param: &Tensor<B, 2>,
    //     w_numerator_sensory: &Tensor<B, 2>,
    //     w_denominator_sensory: &Tensor<B, 2>,
    //     cm_t: &Tensor<B, 2>,
    //     delta_t: f32,
    // ) -> (Tensor<B, 2>, Tensor<B, 2>) {

    //     // gating variable dn/dt = (n∞(V) - n) / (τn + ε)
    //     let n_inf_slope = self.n_inf_slope.val().clone().unsqueeze_dim(0);
    //     let n_inf_thresh = self.n_inf_thresh.val().clone().unsqueeze_dim(0);
    //     let tau_n = self.make_positive_1d(self.tau_n.val().clone()).unsqueeze_dim(0);

    //     // n∞(V)
    //     let n_inf = activation::sigmoid(n_inf_slope * (v_pre.clone() - n_inf_thresh));

    //     // (n∞(V) - n) / (τn + ε)
    //     let dn = (n_inf - n_pre.clone()) / (tau_n + self.epsilon);

    //     let n = n_pre + dn * delta_t;

    //     // s(v)
    //     let mut w_activation = self.sigmoid(
    //         v_pre.clone(),
    //         self.mu.val().clone(),
    //         self.sigma.val().clone(),
    //     );

    //     // wiring sparsity mask
    //     if let Some(mask) = &self.sparsity_mask {
    //         w_activation = w_activation * mask.clone().unsqueeze_dim(0);
    //     }

    //     let n_expanded = n.clone().unsqueeze_dim(2);  // [batch_size, state_size, 1]
    //     let w_param_expanded = w_param.clone().unsqueeze_dim(0);  // [1, state_size, state_size]

    //     // Σ(wij * s(V) * n) - sensory
    //     w_activation = w_activation * w_param_expanded * n_expanded;
    
    //     // Σ(wij * s(V) * n * Erev) - - sensory
    //     let rev_activation = w_activation.clone() * self.erev.val().clone().unsqueeze_dim(0);

    //     // Σ(wij * s(V) * n * Erev)
    //     let w_numerator = rev_activation.sum_dim(1).squeeze_dims(&[1]) + w_numerator_sensory.clone();
    //     // Σ(wij * s(V) * n)
    //     let w_denominator = w_activation.sum_dim(1).squeeze_dims(&[1]) + w_denominator_sensory.clone();

    //     let gleak = self.make_positive_1d(self.gleak.val().clone()).unsqueeze_dim(0);
    //     let vleak = self.vleak.val().clone().unsqueeze_dim(0);

    //     // ((gleak * Vleak) + Σ(wij * s(V) * n * Erev))
    //     let numerator = (gleak.clone() * vleak) + w_numerator;
    //     // (gleak + Σ(wij * s(V) * n)
    //     let denominator = gleak + w_denominator;
        
    //     // [((gleak * Vleak) + Σ(wij * s(V) * n * Erev)) / (gleak + Σ(wij * s(V) * n) + ε) - V]
    //     let mut dv = (numerator / (denominator + self.epsilon)) - v_pre.clone();

    //     // 1/(Cm + ε) * [((gleak * Vleak) + Σ(wij * s(V) * n * Erev)) / (gleak + Σ(wij * s(V) * n) + ε) - V]
    //     dv = dv / (cm_t.clone() + self.epsilon);

    //     let v = v_pre + dv * delta_t;

    //     (v, n)
    // }

    // BRRRRR
    fn compute_states(
        &self,
        v_pre: Tensor<B, 2>,
        n_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>, 
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
        delta_t: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Compute gating variable n
        let n_slope_thresh = self.n_inf_slope.val().clone().unsqueeze_dim(0) * 
            (v_pre.clone() - self.n_inf_thresh.val().clone().unsqueeze_dim(0));
        let n_inf = activation::sigmoid(n_slope_thresh);
        let tau_n_eps = self.make_positive_1d(self.tau_n.val().clone()).unsqueeze_dim(0) + self.epsilon;
        let n = n_pre.clone() + ((n_inf - n_pre) / tau_n_eps) * delta_t;
    
        // Combine sigmoid and masking
        let w_activation = {
            let sig = self.sigmoid(v_pre.clone(), self.mu.val().clone(), self.sigma.val().clone());
            if let Some(mask) = &self.sparsity_mask {
                sig * mask.clone().unsqueeze_dim(0)
            } else {
                sig
            }
        };
    
        // Compute numerator and denominator together
        let (w_num, w_den) = {
            let w_act = w_activation * w_param.clone().unsqueeze_dim(0) * n.clone().unsqueeze_dim(2);
            let num = (w_act.clone() * self.erev.val().clone().unsqueeze_dim(0))
                .sum_dim(1).squeeze_dims(&[1]) + w_numerator_sensory.clone();
            let den = w_act.sum_dim(1).squeeze_dims(&[1]) + w_denominator_sensory.clone();
            (num, den)
        };
    
        // Final voltage computation
        let gleak = self.make_positive_1d(self.gleak.val().clone()).unsqueeze_dim(0);
        let v = v_pre.clone() + (((gleak.clone() * self.vleak.val().clone().unsqueeze_dim(0) + w_num) / 
            (gleak + w_den + self.epsilon) - v_pre) / 
            (cm_t.clone() + self.epsilon)) * delta_t;
    
        (v, n)
    }

    fn euler_step(
        &self,
        v_pre: Tensor<B, 2>,
        n_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>,
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
        delta_t: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (v, n) = self.compute_states(
            v_pre.clone(),
            n_pre.clone(),
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
            delta_t
        );
        (v, n)
    }

    fn forward_fixed(
        &self,
        inputs: Tensor<B, 2>,  // [batch_size, input_size]
        state: (Tensor<B, 2>, Tensor<B, 2>), // (voltage_state, gating_state): ([batch_size, state_size],[batch_size, state_size])
        elapsed_time: f32,
        method: IntegrationMethod,
        _tol: Option<f32>,
    ) -> (Tensor<B, 2>, (Tensor<B, 2>, Tensor<B, 2>)) { // (output, new_state)

        let inputs = self.map_inputs(inputs);
        let (mut v_pre, mut n_pre) = state;

        let (w_numerator_sensory, w_denominator_sensory) = self.compute_sensory_effects(&inputs);

        let delta_t = elapsed_time / self.ode_unfolds as f32;
        let cm_t = self.make_positive_1d(self.cm.val().clone()) / delta_t;
        let cm_t = cm_t.unsqueeze_dim(0);
        let w_param = self.make_positive_2d(self.w.val().clone());

        for _ in 0..self.ode_unfolds {
            (v_pre, n_pre) = match method {
                IntegrationMethod::Euler => self.euler_step(
                    v_pre.clone(),
                    n_pre.clone(),
                    &w_param,
                    &w_numerator_sensory,
                    &w_denominator_sensory,
                    &cm_t,
                    delta_t,
                ),
                // _ => (v_pre.clone(), n_pre.clone()),
            };
        }

        let outputs = self.map_outputs(v_pre.clone());

        (outputs, (v_pre, n_pre))
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        state: (Tensor<B, 2>, Tensor<B, 2>), // (voltage_state, gating_state)
        // state: Tensor<B, 2>,
        elapsed_time: f32,
        tol: Option<f32>,
    ) -> (Tensor<B, 2>, (Tensor<B, 2>, Tensor<B, 2>)) {
        match self.integration_method {
            IntegrationMethod::Euler => {
                self.forward_fixed(inputs, state, elapsed_time, IntegrationMethod::Euler, tol)
            },
            // IntegrationMethod::CashKarp => {
            //     self.forward_adaptive(inputs, state, elapsed_time, tol, IntegrationMethod::CashKarp)
            // },
            // IntegrationMethod::BackwardEuler => {
            //     self.forward_adaptive(inputs, state, elapsed_time, tol, IntegrationMethod::BackwardEuler)
            // },
        }
    }
}