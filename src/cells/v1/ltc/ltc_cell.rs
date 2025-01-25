use burn::{
    config::Config,
    module::{
        Module, Param
    },
    tensor::{
        backend::Backend,
        activation, Distribution, Tensor, ElementConversion,
    },
};

use crate::wiring::v1::base::WiringImpl;

const INIT_RANGES: [(f32, f32); 9] = [
    (0.001, 1.0),  // gleak
    (-0.2, 0.2),   // vleak
    (0.4, 0.6),    // cm
    (0.001, 1.0),  // w
    (3.0, 8.0),    // sigma
    (0.3, 0.8),    // mu
    (0.001, 1.0),  // sensory_w
    (3.0, 8.0),    // sensory_sigma
    (0.3, 0.8),    // sensory_mu
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
    CashKarp,
    BackwardEuler,
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
        wiring: WiringImpl<B>,
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
        wiring: WiringImpl<B>,
        motor_size: usize,
    ) -> LTCCell<B> {
        let state_size = wiring.units();
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
            "cash_karp" => IntegrationMethod::CashKarp,
            "backward_euler" => IntegrationMethod::BackwardEuler,
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
            sensory_erev: Param::from_tensor(wiring.sensory_erev_initializer()
                .unwrap_or_else(|| Tensor::zeros([input_size, state_size], device))),

            input_w,
            input_b,
            output_w,
            output_b,

            sparsity_mask: Some(wiring.adjacency_matrix().clone()),
            sensory_sparsity_mask: wiring.sensory_adjacency_matrix().cloned().map(|m| m),

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

    fn sigmoid(
        &self,
        v_pre: Tensor<B, 2>,
        mu: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let v_pre = v_pre.unsqueeze_dim(2);
        let mu = mu.unsqueeze_dim(0);
        let sigma = sigma.unsqueeze_dim(0);
        let mues = v_pre - mu;
        let x = sigma * mues;
        activation::sigmoid(x)
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

    fn compute_sensory_effects(
        &self,
        inputs: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let sensory_w = self.make_positive_2d(self.sensory_w.val().clone());
        let sensory_activation = self.sigmoid(
            inputs.clone(),
            self.sensory_mu.val().clone(),
            self.sensory_sigma.val().clone(),
        );

        let mut sensory_w_activation = sensory_activation.clone()
            * sensory_w.unsqueeze_dim(0);

        if let Some(mask) = &self.sensory_sparsity_mask {
            sensory_w_activation = sensory_w_activation
                * mask.clone().unsqueeze_dim(0);
        }

        let sensory_rev_activation = sensory_w_activation.clone()
            * self.sensory_erev.val().clone().unsqueeze_dim(0);

        let w_numerator_sensory = sensory_rev_activation
            .sum_dim(1)
            .squeeze_dims(&[1]);
        let w_denominator_sensory = sensory_w_activation
            .sum_dim(1)
            .squeeze_dims(&[1]);

        (w_numerator_sensory, w_denominator_sensory)
    }

    fn compute_dv_dt(
        &self,
        v_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>,
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let w_activation = self.sigmoid(
            v_pre.clone(),
            self.mu.val().clone(),
            self.sigma.val().clone(),
        );
        let mut w_activation = w_activation * w_param.clone().unsqueeze_dim(0);

        if let Some(mask) = &self.sparsity_mask {
            w_activation = w_activation * mask.clone().unsqueeze_dim(0);
        }

        let rev_activation = w_activation.clone() * self.erev.val().clone().unsqueeze_dim(0);

        let w_numerator = rev_activation.sum_dim(1).squeeze_dims(&[1]) + w_numerator_sensory.clone();
        let w_denominator = w_activation.sum_dim(1).squeeze_dims(&[1]) + w_denominator_sensory.clone();

        let gleak = self.make_positive_1d(self.gleak.val().clone()).unsqueeze_dim(0);
        let vleak = self.vleak.val().clone().unsqueeze_dim(0);

        let numerator = (gleak.clone() * vleak) + w_numerator;
        let denominator = gleak + w_denominator;
        
        let dv = (numerator / (denominator + self.epsilon)) - v_pre;
        dv / (cm_t.clone() + self.epsilon)
    }

    fn euler_step(
        &self,
        v_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>,
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
        delta_t: f32,
    ) -> Tensor<B, 2> {
        let dv_dt = self.compute_dv_dt(
            v_pre.clone(),
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );
        v_pre + dv_dt * delta_t
    }

    fn cash_karp_step(
        &self,
        v_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>,
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
        delta_t: f32,
        tol: f32,
    ) -> (Tensor<B, 2>, f32, bool) {

        // a coefficients
        let a = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0, 0.0],
            [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0, 0.0],
            [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0, 0.0],
        ];

        // b coefficients for 5th order estimate
        let b = [
            37.0 / 378.0,
            0.0,
            250.0 / 621.0,
            125.0 / 594.0,
            0.0,
            512.0 / 1771.0,
        ];

        // b_star coefficients for 4th order estimate
        let b_star = [
            2825.0 / 27648.0,
            0.0,
            18575.0 / 48384.0,
            13525.0 / 55296.0,
            -277.0 / 14336.0,
            1.0 / 4.0,
        ];

        // Compute k1 to k6
        let k1 = self.compute_dv_dt(
            v_pre.clone(),
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        let k2 = self.compute_dv_dt(
            v_pre.clone() + k1.clone() * a[1][0] * delta_t,
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        let k3 = self.compute_dv_dt(
            v_pre.clone()
                + (k1.clone() * a[2][0] + k2.clone() * a[2][1]) * delta_t,
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        let k4 = self.compute_dv_dt(
            v_pre.clone()
                + (k1.clone() * a[3][0]
                    + k2.clone() * a[3][1]
                    + k3.clone() * a[3][2])
                    * delta_t,
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        let k5 = self.compute_dv_dt(
            v_pre.clone()
                + (k1.clone() * a[4][0]
                    + k2.clone() * a[4][1]
                    + k3.clone() * a[4][2]
                    + k4.clone() * a[4][3])
                    * delta_t,
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        let k6 = self.compute_dv_dt(
            v_pre.clone()
                + (k1.clone() * a[5][0]
                    + k2.clone() * a[5][1]
                    + k3.clone() * a[5][2]
                    + k4.clone() * a[5][3]
                    + k5.clone() * a[5][4])
                    * delta_t,
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
        );

        // 5th order estimate
        let mut y_5th = v_pre.clone();
        for (ki, bi) in [
            (k1.clone(), b[0]),
            (k2.clone(), b[1]),
            (k3.clone(), b[2]),
            (k4.clone(), b[3]),
            (k5.clone(), b[4]),
            (k6.clone(), b[5]),
        ].iter() {
            y_5th = y_5th + (*ki).clone() * (*bi * delta_t);
        }

        let mut y_4th = v_pre.clone();
        for (ki, bi_star) in [
            (k1.clone(), b_star[0]),
            (k2.clone(), b_star[1]),
            (k3.clone(), b_star[2]),
            (k4.clone(), b_star[3]),
            (k5.clone(), b_star[4]),
            (k6.clone(), b_star[5]),
        ].iter() {
            y_4th = y_4th + (*ki).clone() * (*bi_star * delta_t);
        }

        let error = (y_5th.clone() - y_4th.clone()).abs();
        let max_error = error.max().mean().into_scalar().elem::<f32>();
        
        let safety = 0.9;
        let exponent = 0.2;
        let scale = (tol / (max_error + 1e-10)).powf(exponent) * safety;
        let new_delta_t = delta_t * scale.clamp(0.1, 5.0);

        let accept_step = max_error <= tol;

        if accept_step {
            (y_5th, new_delta_t, true)
        } else {
            (y_4th, new_delta_t, false)
        }
    }

    fn backward_euler_step(
        &self,
        v_pre: Tensor<B, 2>,
        w_param: &Tensor<B, 2>,
        w_numerator_sensory: &Tensor<B, 2>,
        w_denominator_sensory: &Tensor<B, 2>,
        cm_t: &Tensor<B, 2>,
        delta_t: f32,
        tol: f32,
        max_iter: usize,
    ) -> (Tensor<B, 2>, f32, bool) {
        let mut v_new = self.euler_step(
            v_pre.clone(),
            w_param,
            w_numerator_sensory,
            w_denominator_sensory,
            cm_t,
            delta_t,
        );
        
        for _ in 0..max_iter {
            let f_new = self.compute_dv_dt(
                v_new.clone(),
                w_param,
                w_numerator_sensory,
                w_denominator_sensory,
                cm_t,
            );
            let v_guess = v_pre.clone() + f_new.clone() * delta_t;
            let error = (v_guess.clone() - v_new.clone()).abs();
            let max_error = error.max().into_scalar().elem::<f32>();
            
            if max_error < tol {
                return (v_guess, delta_t, true);
            }
            
            v_new = v_guess;
        }
        
        (v_new, delta_t / 2.0, false)
    }

    fn forward_fixed(
        &self,
        inputs: Tensor<B, 2>,  // [batch_size, input_size]
        state: Tensor<B, 2>,   // [batch_size, state_size]
        elapsed_time: f32,
        method: IntegrationMethod,
        _tol: Option<f32>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) { // (output, new_state)

        let inputs = self.map_inputs(inputs);
        let mut v_pre = state;

        let (w_numerator_sensory, w_denominator_sensory) = self.compute_sensory_effects(&inputs);

        let delta_t = elapsed_time / self.ode_unfolds as f32;
        let cm_t = self.make_positive_1d(self.cm.val().clone()) / delta_t;
        let cm_t = cm_t.unsqueeze_dim(0);
        let w_param = self.make_positive_2d(self.w.val().clone());

        for _ in 0..self.ode_unfolds {
            v_pre = match method {
                IntegrationMethod::Euler => self.euler_step(
                    v_pre.clone(),
                    &w_param,
                    &w_numerator_sensory,
                    &w_denominator_sensory,
                    &cm_t,
                    delta_t,
                ),
                _ => v_pre.clone(),
            };
        }

        let outputs = self.map_outputs(v_pre.clone());
        (outputs, v_pre)
    }

    pub fn forward_adaptive(
        &self,
        inputs: Tensor<B, 2>,
        state: Tensor<B, 2>,
        elapsed_time: f32,
        tol: Option<f32>,
        method: IntegrationMethod,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {

        let inputs = self.map_inputs(inputs);
        let mut v_pre = state;

        let (w_numerator_sensory, w_denominator_sensory) = self.compute_sensory_effects(&inputs);

        let mut remaining_time = elapsed_time;
        let mut delta_t = elapsed_time / self.ode_unfolds as f32;

        let cm_t = self.make_positive_1d(self.cm.val().clone()) / elapsed_time;
        let cm_t = cm_t.unsqueeze_dim(0);
        let w_param = self.make_positive_2d(self.w.val().clone());

        while remaining_time > 0.0 {
            if delta_t > remaining_time {
                delta_t = remaining_time;
            }

            let (new_v, new_delta_t, accept) = match method {
                IntegrationMethod::CashKarp => {
                    self.cash_karp_step(
                        v_pre.clone(),
                        &w_param,
                        &w_numerator_sensory,
                        &w_denominator_sensory,
                        &cm_t,
                        delta_t,
                        tol.unwrap_or(self.tolerance),
                    )
                },
                IntegrationMethod::BackwardEuler => {
                    self.backward_euler_step(
                        v_pre.clone(),
                        &w_param,
                        &w_numerator_sensory,
                        &w_denominator_sensory,
                        &cm_t,
                        delta_t,
                        tol.unwrap_or(self.tolerance),
                        10, // Example max_iter
                    )
                },
                _ => (v_pre.clone(), delta_t, true),
            };

            if accept {
                v_pre = new_v;
                remaining_time -= delta_t;
                delta_t = new_delta_t;
            } else {
                delta_t = new_delta_t;
            }
        }

        let outputs = self.map_outputs(v_pre.clone());
        (outputs, v_pre)
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        state: Tensor<B, 2>,
        elapsed_time: f32,
        tol: Option<f32>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self.integration_method {
            IntegrationMethod::Euler => {
                self.forward_fixed(inputs, state, elapsed_time, IntegrationMethod::Euler, tol)
            },
            IntegrationMethod::CashKarp => {
                self.forward_adaptive(inputs, state, elapsed_time, tol, IntegrationMethod::CashKarp)
            },
            IntegrationMethod::BackwardEuler => {
                self.forward_adaptive(inputs, state, elapsed_time, tol, IntegrationMethod::BackwardEuler)
            },
        }
    }
}