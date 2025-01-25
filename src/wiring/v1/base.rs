use burn::prelude::*;
use burn::tensor::backend::Backend;
use rand::{
    rngs::StdRng, SeedableRng
};
use thiserror::Error;
use plotters::prelude::*;
use std::collections::HashMap;
use plotters::prelude::BitMapBackend;
use petgraph::{
    visit::EdgeRef, graph::DiGraph
}; 
use crate::wiring::v1::{
    ncp::NCP,
    auto_ncp::AutoNCP,
    fully_connected::FullyConnected,
    random::Random,
};

// ########################################################################################################
// Utils

#[derive(Debug)]
struct NodeInfo {
    id: usize,
    neuron_type: NeuronType,
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    Motor,
    Command,
    Inter,
    Sensory,
}

impl NeuronType {
    fn to_str(&self) -> &'static str {
        match self {
            NeuronType::Motor => "motor",
            NeuronType::Command => "command",
            NeuronType::Inter => "inter",
            NeuronType::Sensory => "sensory",
        }
    }
}

// ########################################################################################################
// Errors

#[derive(Error, Debug)]
pub enum WiringError {
    #[error("Cannot add synapse from {0} when network has only {1} units")]
    InvalidSourceNeuron(usize, usize),
    #[error("Cannot add synapse to {0} when network has only {1} units")]
    InvalidDestinationNeuron(usize, usize),
    #[error("Cannot add synapse with polarity {0} (expected -1 or +1)")]
    InvalidPolarity(i32),
    #[error("Cannot add sensory synapses before build() has been called")]
    NetworkNotBuilt,
    #[error("Cannot add sensory synapse from {0} when input has only {1} features")]
    InvalidSensoryNeuron(usize, usize),
    #[error("Conflicting input dimensions: expected {0}, got {1}")]
    ConflictingDimensions(usize, usize),
    #[error("No neurons available")]
    NoNeuronsAvailable,
    #[error("Invalid Parameters")]
    InvalidParameter,
    #[error("Layer organization error: total neurons mismatch")]
    LayerOrganizationError,
    #[error("{0}")]
    InitializationError(String),
}

// ########################################################################################################
// Wiring

#[derive(Debug, Clone)]
pub struct Wiring<B: Backend> {
    pub units: usize,

    /// internal connectivity tensor (neuron-to-neuron)
    pub adjacency_matrix: Tensor<B, 2>,
    /// input connectivity tneosr (input-to-neuron) 
    pub sensory_adjacency_matrix: Option<Tensor<B, 2>>,

    pub input_dim: Option<usize>,
    pub output_dim: Option<usize>,

    pub device: B::Device,
    /// neuron types
    pub neuron_types: Vec<NeuronType>,
    /// layer org
    pub motor_neurons: Vec<usize>,
    pub command_neurons: Vec<usize>,
    pub inter_neurons: Vec<usize>,

    pub seed: u64,
    pub rng: StdRng,
}

impl<B: Backend> Wiring<B> {
    pub fn new(units: usize, device: B::Device, seed: u64) -> Self {
        let adjacency_matrix = Tensor::zeros([units, units], &device);
        let neuron_types = vec![NeuronType::Inter; units];
        let rng = StdRng::seed_from_u64(seed);

        Self {
            units,
            adjacency_matrix,
            sensory_adjacency_matrix: None,
            input_dim: None,
            output_dim: None,
            device,
            neuron_types,
            motor_neurons: Vec::new(),
            command_neurons: Vec::new(),
            inter_neurons: Vec::new(),
            seed,
            rng,
        }
    }

    pub fn num_layers(&self) -> usize {
        3 // Motor, Command, Inter layers
    }

    pub fn get_neurons_of_layer(&self, layer_id: usize) -> Vec<usize> {
        match layer_id {
            0 => self.inter_neurons.clone(),
            1 => self.command_neurons.clone(),
            2 => self.motor_neurons.clone(),
            _ => Vec::new(),
        }
    }

    pub fn organize_layers(
        &mut self,
        motor_count: usize,
        command_count: usize,
        _inter_count: usize,
    ) -> Result<(), WiringError> {

        self.motor_neurons = (0..motor_count).collect();
        self.command_neurons = (motor_count..motor_count+command_count).collect();
        self.inter_neurons = (motor_count+command_count..self.units).collect();

        self.neuron_types = vec![NeuronType::Inter; self.units];
        
        for i in &self.motor_neurons {
            self.neuron_types[*i] = NeuronType::Motor;
        }
        for i in &self.command_neurons {
            self.neuron_types[*i] = NeuronType::Command;
        }

        Ok(())
    }

    pub fn get_type_of_neuron(&self, neuron_id: usize) -> NeuronType {
        self.neuron_types[neuron_id]
    }

    pub fn is_built(&self) -> bool {
        self.input_dim.is_some()
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {
        if let Some(existing_dim) = self.input_dim {
            if existing_dim != input_dim {
                return Err(WiringError::ConflictingDimensions(existing_dim, input_dim));
            }
        } else {
            self.set_input_dim(input_dim);
        }
        Ok(())
    }

    pub fn set_input_dim(&mut self, input_dim: usize) {
        self.input_dim = Some(input_dim);
        self.sensory_adjacency_matrix = Some(Tensor::zeros([input_dim, self.units], &self.device));
    }

    pub fn set_output_dim(&mut self, output_dim: usize) {
        self.output_dim = Some(output_dim);
    }

    pub fn add_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
    ) -> Result<(), WiringError> {
        if src >= self.units {
            return Err(WiringError::InvalidSourceNeuron(src, self.units));
        }
        if dest >= self.units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        let value = Tensor::from_data([[polarity as f32]], &self.device);
        self.adjacency_matrix = self.adjacency_matrix.clone()
            .slice_assign([src..src + 1, dest..dest + 1], value);

        Ok(())
    }

    pub fn add_sensory_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
    ) -> Result<(), WiringError> {
        let input_dim = self.input_dim.ok_or(WiringError::NetworkNotBuilt)?;

        if src >= input_dim {
            return Err(WiringError::InvalidSensoryNeuron(src, input_dim));
        }
        if dest >= self.units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        if let Some(ref mut matrix) = self.sensory_adjacency_matrix {
            let value = Tensor::from_data([[polarity as f32]], &self.device);
            *matrix = matrix.clone()
                .slice_assign([src..src + 1, dest..dest + 1], value);
        }

        Ok(())
    }

    pub fn synapse_count(&self) -> usize {
        self.adjacency_matrix.clone().abs().sum().into_scalar().elem::<f32>() as usize
    }

    pub fn sensory_synapse_count(&self) -> usize {
        self.sensory_adjacency_matrix
            .as_ref()
            .map(|m| m.clone().abs().sum().into_scalar().elem::<f32>() as usize)
            .unwrap_or(0)
    }

    pub fn erev_initializer(&self) -> Tensor<B, 2> {
        self.adjacency_matrix.clone()
    }

    pub fn sensory_erev_initializer(&self) -> Option<Tensor<B, 2>> {
        self.sensory_adjacency_matrix.clone()
    }

    pub fn print_wiring(&self) {
        println!("\nWiring Information:");
        println!("==================");
        
        println!("\nLayer Organization:");
        println!("Motor neurons: {:?}", self.motor_neurons);
        println!("Command neurons: {:?}", self.command_neurons);
        println!("Inter neurons: {:?}", self.inter_neurons);
        
        println!("\nConnectivity Statistics:");
        println!("Total neurons: {}", self.units);
        println!("Input features: {:?}", self.input_dim);
        println!("Output neurons: {:?}", self.output_dim);
        println!("Internal synapses: {}", self.synapse_count());
        println!("Sensory synapses: {}", self.sensory_synapse_count());

        println!("\nInternal Connections:");
        self.print_internal_connections();
        
        println!("\nSensory Connections:");
        self.print_sensory_connections();
    }

    fn print_internal_connections(&self) {
        for src in 0..self.units {
            for dest in 0..self.units {
                let weight = self.adjacency_matrix.clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar()
                    .elem::<f32>();
                if weight != 0.0 {
                    let polarity = if weight > 0.0 { "+" } else { "-" };
                    println!("Neuron {} ({:?}) --({})--> Neuron {} ({:?})",
                             src, self.neuron_types[src], polarity, dest, self.neuron_types[dest]);
                }
            }
        }
    }

    fn print_sensory_connections(&self) {
        if let Some(ref sensory_matrix) = self.sensory_adjacency_matrix {
            let input_dim = self.input_dim.unwrap_or(0);
            for src in 0..input_dim {
                for dest in 0..self.units {
                    let weight = sensory_matrix.clone()
                        .slice([src..src + 1, dest..dest + 1])
                        .into_scalar()
                        .elem::<f32>();
                    if weight != 0.0 {
                        let polarity = if weight > 0.0 { "+" } else { "-" };
                        println!("Input {} --({})--> Neuron {} ({:?})",
                                 src, polarity, dest, self.neuron_types[dest]);
                    }
                }
            }
        }
    }

    pub fn draw_graph(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Canvas settings
        let width = 800u32;
        let height = 600u32;
        let margin = 50.0f64;
        
        // Create drawing area
        let root = BitMapBackend::new(filename, (width, height))
            .into_drawing_area();
        root.fill(&WHITE)?;
    
        // Create basic mapping between data and pixels
        let mut chart_builder = ChartBuilder::on(&root)
            .margin(margin as i32)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..(width as f64),
                0f64..(height as f64)
            )?;
    
        // Configure chart
        chart_builder
            .configure_mesh()
            .disable_mesh()
            .disable_axes()
            .draw()?;
    
        // Calculate layer positions (vertical centers)
        let layer_width = (width as f64 - 2.0 * margin) / 4.0;
        let sensory_x = margin;
        let inter_x = margin + layer_width;
        let command_x = margin + 2.0 * layer_width;
        let motor_x = margin + 3.0 * layer_width;
    
        // Sinusoidal pattern parameters
        let wave_amplitude = layer_width * 0.25; // Horizontal deviation from center
        // let vertical_spacing = (height as f64 - 2.0 * margin) / 10.0; // Base vertical spacing between neurons
    
        // Create graph
        let mut graph = DiGraph::<NodeInfo, f32>::new();
        let mut node_indices = HashMap::new();
    
        // Helper function to calculate sinusoidal position
        let calculate_sin_pos = |idx: usize, total: usize, center_x: f64| -> (f64, f64) {
            if total == 1 {
                return (center_x, height as f64 / 2.0);
            }
            
            // Calculate base vertical position
            let vertical_range = height as f64 - 2.0 * margin;
            let y_base = margin + (vertical_range * idx as f64) / (total.max(2) - 1) as f64;
            
            // Calculate horizontal offset using alternating pattern
            let x_offset = if idx % 2 == 0 { wave_amplitude } else { -wave_amplitude };
            
            (center_x + x_offset, y_base)
        };
    
        // Add sensory nodes
        let input_dim = self.input_dim.unwrap_or(0);
        for i in 0..input_dim {
            let (x, y) = calculate_sin_pos(i, input_dim, sensory_x);
            let node = NodeInfo {
                id: i,
                neuron_type: NeuronType::Sensory,
                x,
                y,
            };
            let idx = graph.add_node(node);
            node_indices.insert(("sensory", i), idx);
        }
    
        // Add inter nodes
        for (i, &neuron_id) in self.inter_neurons.iter().enumerate() {
            let (x, y) = calculate_sin_pos(i, self.inter_neurons.len(), inter_x);
            let node = NodeInfo {
                id: neuron_id,
                neuron_type: NeuronType::Inter,
                x,
                y,
            };
            let idx = graph.add_node(node);
            node_indices.insert(("inter", neuron_id), idx);
        }
    
        // Add command nodes
        for (i, &neuron_id) in self.command_neurons.iter().enumerate() {
            let (x, y) = calculate_sin_pos(i, self.command_neurons.len(), command_x);
            let node = NodeInfo {
                id: neuron_id,
                neuron_type: NeuronType::Command,
                x,
                y,
            };
            let idx = graph.add_node(node);
            node_indices.insert(("command", neuron_id), idx);
        }
    
        // Add motor nodes
        for (i, &neuron_id) in self.motor_neurons.iter().enumerate() {
            let (x, y) = calculate_sin_pos(i, self.motor_neurons.len(), motor_x);
            let node = NodeInfo {
                id: neuron_id,
                neuron_type: NeuronType::Motor,
                x,
                y,
            };
            let idx = graph.add_node(node);
            node_indices.insert(("motor", neuron_id), idx);
        }
    
        // Add edges based on internal adjacency matrix
        for src in 0..self.units {
            for dest in 0..self.units {
                let weight = self.adjacency_matrix.clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar()
                    .elem::<f32>();
                if weight != 0.0 {
                    let src_type = self.neuron_types[src];
                    let dest_type = self.neuron_types[dest];
                    let src_key = (src_type.to_str(), src);
                    let dest_key = (dest_type.to_str(), dest);
                    let src_idx = *node_indices.get(&src_key).expect("Source node not found");
                    let dest_idx = *node_indices.get(&dest_key).expect("Destination node not found");
                    graph.add_edge(src_idx, dest_idx, weight);
                }
            }
        }
    
        // Add edges based on sensory adjacency matrix
        if let Some(ref sensory_matrix) = self.sensory_adjacency_matrix {
            let input_dim = self.input_dim.unwrap_or(0);
            for src in 0..input_dim {
                for dest in 0..self.units {
                    let weight = sensory_matrix.clone()
                        .slice([src..src + 1, dest..dest + 1])
                        .into_scalar()
                        .elem::<f32>();
                    if weight != 0.0 {
                        let dest_type = self.neuron_types[dest];
                        let src_key = ("sensory", src);
                        let dest_key = (dest_type.to_str(), dest);
                        let src_idx = *node_indices.get(&src_key).expect("Source node not found");
                        let dest_idx = *node_indices.get(&dest_key).expect("Destination node not found");
                        graph.add_edge(src_idx, dest_idx, weight);
                    }
                }
            }
        }
    
        // Draw edges with curved paths
        for edge in graph.edge_references() {
            let src_idx = edge.source();
            let dest_idx = edge.target();
            let weight = edge.weight();
    
            let src_node: &NodeInfo = &graph[src_idx];
            let dest_node: &NodeInfo = &graph[dest_idx];
    
            let color = if *weight > 0.0 { &GREEN } else { &RED };
    
            // Calculate control points for curved edges
            let ctrl_point_1 = (
                src_node.x + (dest_node.x - src_node.x) / 3.0,
                src_node.y
            );
            let ctrl_point_2 = (
                src_node.x + 2.0 * (dest_node.x - src_node.x) / 3.0,
                dest_node.y
            );
    
            // Draw curved edge using cubic Bézier curve approximation
            let steps = 50;
            let curve_points: Vec<(f64, f64)> = (0..=steps)
                .map(|i| {
                    let t = i as f64 / steps as f64;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;
                    
                    let x = src_node.x * mt3 + 
                           3.0 * ctrl_point_1.0 * mt2 * t +
                           3.0 * ctrl_point_2.0 * mt * t2 +
                           dest_node.x * t3;
                           
                    let y = src_node.y * mt3 +
                           3.0 * ctrl_point_1.1 * mt2 * t +
                           3.0 * ctrl_point_2.1 * mt * t2 +
                           dest_node.y * t3;
                           
                    (x, y)
                })
                .collect();
    
            chart_builder.draw_series(LineSeries::new(
                curve_points.clone(),
                color.stroke_width(1),
            ))?;
    
            // Add arrow heads
            let arrow_len = 10.0;
            let arrow_width = 6.0;
            
            // Calculate the direction at the end of the curve
            let last_segment = curve_points.windows(2).last().unwrap();
            let dx = last_segment[1].0 - last_segment[0].0;
            let dy = last_segment[1].1 - last_segment[0].1;
            let angle = dy.atan2(dx);
            
            // Calculate arrow head points
            let tip_x = dest_node.x - 8.0 * dx.signum();
            let tip_y = dest_node.y - 8.0 * dy.signum();
            let left_x = tip_x - arrow_len * angle.cos() + arrow_width * (angle + std::f64::consts::PI/2.0).cos();
            let left_y = tip_y - arrow_len * angle.sin() + arrow_width * (angle + std::f64::consts::PI/2.0).sin();
            let right_x = tip_x - arrow_len * angle.cos() + arrow_width * (angle - std::f64::consts::PI/2.0).cos();
            let right_y = tip_y - arrow_len * angle.sin() + arrow_width * (angle - std::f64::consts::PI/2.0).sin();
    
            chart_builder.draw_series(std::iter::once(Polygon::new(
                vec![
                    (tip_x, tip_y),
                    (left_x, left_y),
                    (right_x, right_y),
                ],
                color.filled(),
            )))?;
        }
    
        // Draw nodes
        for node_idx in graph.node_indices() {
            let node: &NodeInfo = &graph[node_idx];
            let color = match node.neuron_type {
                NeuronType::Sensory => &BLUE,
                NeuronType::Inter => &CYAN,
                NeuronType::Command => &MAGENTA,
                NeuronType::Motor => &YELLOW,
            };
    
            // Draw node circle
            chart_builder.draw_series(std::iter::once(Circle::new(
                (node.x, node.y),
                5,
                color.filled(),
            )))?;
    
            // Draw node label
            chart_builder.draw_series(std::iter::once(Text::new(
                format!("{}", node.id),
                (node.x, node.y - 15.0),
                ("sans-serif", 15).into_font(),
            )))?;
        }
    
        // Draw legend
        let legend_y = height as f64 - 100.0;
        
        // Node types legend
        let node_types = [
            ("Sensory", &BLUE),
            ("Inter", &CYAN),
            ("Command", &MAGENTA),
            ("Motor", &YELLOW),
        ];
    
        for (i, (label, color)) in node_types.iter().enumerate() {
            let y = legend_y + (i as f64 * 20.0);
            
            // Draw node example
            chart_builder.draw_series(std::iter::once(Circle::new(
                (margin + 10.0, y),
                5,
                color.filled(),
            )))?;
    
            // Draw label
            chart_builder.draw_series(std::iter::once(Text::new(
                *label,
                (margin + 25.0, y),
                ("sans-serif", 12).into_font(),
            )))?;
        }
    
        // Connection types legend
        let center_x = width as f64 / 2.0;
        let connection_types = [
            ("Excitatory", &GREEN),
            ("Inhibitory", &RED),
        ];
    
        for (i, (label, color)) in connection_types.iter().enumerate() {
            let y = legend_y + (i as f64 * 20.0);
            
            // Draw curved line example with arrow
            let line_start_x = center_x;
            let line_end_x = center_x + 30.0;
            let control1_x = line_start_x + 10.0;
            let control2_x = line_end_x - 10.0;
            
            let curve_points: Vec<(f64, f64)> = (0..=20)
                .map(|i| {
                    let t = i as f64 / 20.0;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;
                    
                    let x = line_start_x * mt3 + 
                           3.0 * control1_x * mt2 * t +
                           3.0 * control2_x * mt * t2 +
                           line_end_x * t3;
                    let y = y;
                    
                    (x, y)
                })
                .collect();
    
            chart_builder.draw_series(LineSeries::new(
                curve_points,
                color.stroke_width(2),
            ))?;
    
            // Add arrow head to legend
            let arrow_len = 8.0;
            let arrow_width = 4.0;
            chart_builder.draw_series(std::iter::once(Polygon::new(
                vec![
                    (line_end_x, y),
                    (line_end_x - arrow_len, y - arrow_width),
                    (line_end_x - arrow_len, y + arrow_width),
                ],
                color.filled(),
            )))?;
    
            // Draw label
            chart_builder.draw_series(std::iter::once(Text::new(
                *label,
                (line_end_x + 15.0, y),
                ("sans-serif", 12).into_font(),
            )))?;
        }
    
        root.present()?;
        Ok(())
    }

}


// ########################################################################################################
// WiringImpl

#[derive(Debug, Clone)]
pub enum WiringImpl<B: Backend> {
    Random(Random<B>),
    FullyConnected(FullyConnected<B>),
    NCP(NCP<B>), 
    AutoNCP(AutoNCP<B>)
}

impl<B: Backend> WiringImpl<B> {
    pub fn units(&self) -> usize {
        match self {
            Self::Random(w) => w.wiring.units,
            Self::FullyConnected(w) => w.wiring.units,
            Self::NCP(w) => w.wiring.units,
            Self::AutoNCP(w) => w.units()
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Self::Random(_w) => "Random".to_string(),
            Self::FullyConnected(_w) => "FullyConnected".to_string(),
            Self::NCP(_w) => "NCP".to_string(),
            Self::AutoNCP(_w) => "AutoNCP".to_string(),
        }
    }

    pub fn output_dim(&self) -> Option<usize> {
        match self {
            Self::Random(w) => w.wiring.output_dim,
            Self::FullyConnected(w) => w.wiring.output_dim,
            Self::NCP(w) => w.wiring.output_dim,
            Self::AutoNCP(w) => Some(w.output_size())
        }
    }
    
    pub fn input_dim(&self) -> Option<usize> {
        match self {
            Self::Random(w) => w.wiring.input_dim,
            Self::FullyConnected(w) => w.wiring.input_dim,
            Self::NCP(w) => w.wiring.input_dim,
            Self::AutoNCP(w) => w.ncp.wiring.input_dim,
        }
    }

    pub fn seed(&self) -> usize {
        match self {
            Self::Random(w) => w.wiring.seed.try_into().unwrap(),
            Self::FullyConnected(w) => w.wiring.seed.try_into().unwrap(),
            Self::NCP(w) => w.wiring.seed.try_into().unwrap(),
            Self::AutoNCP(w) => w.ncp.wiring.seed.try_into().unwrap(),
        }
    }

    pub fn adjacency_matrix(&self) -> &Tensor<B, 2> {
        match self {
            Self::Random(w) => &w.wiring.adjacency_matrix,
            Self::FullyConnected(w) => &w.wiring.adjacency_matrix,
            Self::NCP(w) => &w.wiring.adjacency_matrix,
            Self::AutoNCP(w) => &w.ncp.wiring.adjacency_matrix
        }
    }

    pub fn sensory_adjacency_matrix(&self) -> Option<&Tensor<B, 2>> {
        match self {
            Self::Random(w) => w.wiring.sensory_adjacency_matrix.as_ref(),
            Self::FullyConnected(w) => w.wiring.sensory_adjacency_matrix.as_ref(),
            Self::NCP(w) => w.wiring.sensory_adjacency_matrix.as_ref(),
            Self::AutoNCP(w) => w.ncp.wiring.sensory_adjacency_matrix.as_ref()
        }
    }

    pub fn erev_initializer(&self) -> Tensor<B, 2> {
        match self {
            Self::Random(w) => w.wiring.erev_initializer(),
            Self::FullyConnected(w) => w.wiring.erev_initializer(),
            Self::NCP(w) => w.wiring.erev_initializer(),
            Self::AutoNCP(w) => w.ncp.wiring.erev_initializer()
        }
    }

    pub fn sensory_erev_initializer(&self) -> Option<Tensor<B, 2>> {
        match self {
            Self::Random(w) => w.wiring.sensory_erev_initializer(),
            Self::FullyConnected(w) => w.wiring.sensory_erev_initializer(),
            Self::NCP(w) => w.wiring.sensory_erev_initializer(),
            Self::AutoNCP(w) => w.ncp.wiring.sensory_erev_initializer()
        }
    }

    pub fn build(&mut self, input_dim: usize) -> Result<(), WiringError> {
        match self {
            Self::Random(w) => w.build(input_dim),
            Self::FullyConnected(w) => w.build(input_dim),
            Self::NCP(w) => w.build(input_dim),
            Self::AutoNCP(w) => w.build(input_dim)
        }
    }

    pub fn draw_graph(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Random(w) => w.wiring.draw_graph(filename),
            Self::FullyConnected(w) => w.wiring.draw_graph(filename),
            Self::NCP(w) => w.wiring.draw_graph(filename),
            Self::AutoNCP(w) => w.ncp.wiring.draw_graph(filename),
        }
    }
}