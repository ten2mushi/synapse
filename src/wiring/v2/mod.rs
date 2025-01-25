use burn::prelude::*;
use burn::tensor::backend::Backend;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;
use plotters::prelude::*;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;

// USAGE    ###############################################################################################

// -- A wiring is a network of interconnected layers.
// Each layer is defined by:
//     - a number of units
//     - a definition of outgoing connections (vec<Fanout>)

// See:
// #[derive(Debug, Clone)]
// pub enum Fanout {
//     Forward(f64, f64),              // (sparsity, polarity) // both between 0 and 1
//     Backward(f64, f64),             // (sparsity, polarity)
//     Recurrent(usize, f64),          // (count, polarity)
//     LongRange(usize, f64, f64),     // (target_layer_index, (sparsity, polarity))
//     None,                           // No connections (motor layer)
// }

// Fanout::Forward(sparsity, polarity)
//     - a connection to the next layer
// Fanout::Backward(sparsity, polarity)
//     - a connection to the previous layer
// Fanout::Recurrent(count, polarity)
//     - a reccurent connection inside the layer
// Fanout::LongRange(target_layer_index, (sparsity, polarity))
//     - a connection to another layer targeted by layer_id


// -- to create a wiring:

// let input_size = 2;
// let output_size = 3;
// let model_seed = 42;
// let device == LibTorch;

// -- first define layers:
// let layer_configs = vec![

//     // first layer == sensory layer
//     LayerConfig {
//         num_units: input_size,
//         connections: vec![
//             Fanout::Forward(0.3, 0.8),    // Connect to next layer with 30% sparsity, 80% excitatory
//         ],
//     },

//     LayerConfig {
//         num_units: 7,
//         connections: vec![
//             Fanout::Forward(0.9, 0.8),    // Connect to next layer with 90% sparsity, 80% excitatory
//             Fanout::LongRange(3, 0.9, 0.7), // Connect to layer 3 with 90% sparsity, 70% excitatory
//         ],
//     },
//     LayerConfig {
//         num_units: 4,
//         connections: vec![
//             Fanout::Forward(0.9, 0.8),    // Connect to next layer with 90% sparsity, 80% excitatory
//         ],
//     },
//     LayerConfig {
//         num_units: 3,
//         connections: vec![
//             Fanout::Forward(0.9, 0.8),    // Connect to next layer with 90% sparsity, 80% excitatory
//             Fanout::Recurrent(2, 0.5),    // 2 recurrent connections, 50% excitatory
//         ],
//     },

//     // last layer == motor layer
//     LayerConfig {
//         num_units: output_size,
//         connections: vec![
//             // leave empty, last layer
//         ],
//     },
// ];

// -- then create wiring
// let wiring = Wiring::new(
//     input_size,
//     output_size,
//     layer_configs,
//     model_seed,
//     &device,
// );

// ########################################################################################################

#[derive(Error, Debug)]
pub enum WiringError {
    #[error("Cannot add synapse from {0} when network has only {1} units")]
    InvalidSourceNeuron(usize, usize),
    #[error("Cannot add synapse to {0} when network has only {1} units")]
    InvalidDestinationNeuron(usize, usize),
    #[error("Cannot add synapse with polarity {0} (expected -1 or +1)")]
    InvalidPolarity(i32),
    #[error("Cannot add sensory synapse from {0} when input has only {1} features")]
    InvalidSensoryNeuron(usize, usize),
    #[error("Invalid Parameters")]
    InvalidParameter,
    #[error("Layer organization error: total neurons mismatch")]
    LayerOrganizationError,
    #[error("{0}")]
    InitializationError(String),
}


#[derive(Debug)]
struct NodeInfo {
    id: usize,
    layer_id: usize,
    x: f64,
    y: f64,
}

#[derive(Debug, Clone)]
pub enum Fanout {
    Forward(f64, f64),              // (sparsity, polarity) // both between 0 and 1
    Backward(f64, f64),             // (sparsity, polarity)
    Recurrent(usize, f64),          // (count, polarity)
    LongRange(usize, f64, f64),     // (target_layer_index, (sparsity, polarity))
    None,                           // No connections (motor layer)
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub num_units: usize,
    pub connections: Vec<Fanout>,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub config: LayerConfig,
}

#[derive(Debug)]
struct FanoutPlan {
    _source_id: usize,
    _target_id: usize,
    source_range: Range<usize>,
    target_range: Range<usize>,
    connection_type: FanoutType,
    sensory: bool,
}

#[derive(Debug)]
enum FanoutType {
    Forward { sparsity: f64, polarity: f64 },
    Backward { sparsity: f64, polarity: f64 },
    Recurrent { count: usize, polarity: f64 },
    LongRange { sparsity: f64, polarity: f64 },
}

#[derive(Debug, Clone)]
pub struct Wiring<B: Backend> {
    pub input_size: usize,
    pub output_size: usize,
    pub total_units: usize,
    pub adjacency_matrix: Tensor<B, 2>,
    pub sensory_adjacency_matrix: Tensor<B, 2>,
    pub layers: Vec<Layer>,
    pub layer_map: Vec<usize>,
    pub neuron_map: Vec<Vec<usize>>,
    pub neuron_map_flatten: Vec<usize>,
    pub rng: StdRng,
}

impl<B: Backend> Wiring<B> {
    pub fn new(
        layer_configs: Vec<LayerConfig>,
        seed: u64,
        device: &B::Device,
    ) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::new();
        let mut layer_map = Vec::new();
        let mut neuron_map = Vec::new();
        let input_size = layer_configs[0].num_units;
        let output_size = layer_configs[layer_configs.len() - 1].num_units;
        let sensory_neurons: Vec<usize> = (0..input_size).collect();

        neuron_map.push(sensory_neurons);
        let total_hidden_units: usize = layer_configs.iter().skip(1)
            .map(|config| config.num_units)
            .sum();
        
        let mut current_id = 0;
        for (rev_idx, config) in layer_configs.iter().rev().enumerate() {
            let start_idx = current_id;
            current_id += config.num_units;
            let end_idx = current_id;
            let layer_id = (layer_configs.len() - 1) - rev_idx;
            let layer_neurons: Vec<usize> = (start_idx..end_idx).collect();
            for _ in 0..config.num_units {
                layer_map.insert(0, layer_id);
            }
            layers.insert(0, Layer {
                id: layer_id,
                start_idx,
                end_idx,
                config: config.clone(),
            });
            if layer_id != 0 {
                neuron_map.insert(1, layer_neurons);
            }
        }
        
        let neuron_map_flatten = neuron_map.iter().flatten().copied().collect();

        let mut wiring = Self {
            input_size,
            output_size,
            total_units: total_hidden_units,
            adjacency_matrix: Tensor::zeros([total_hidden_units, total_hidden_units], device),
            sensory_adjacency_matrix: Tensor::zeros([input_size, total_hidden_units], device),
            layers,
            layer_map,
            neuron_map,
            neuron_map_flatten,
            rng,
        };
        
        wiring.build_connections(device).unwrap();
        wiring
    }

    fn build_connections(
        &mut self,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let connection_plans = self.create_connection_plans()?;
        
        for plan in connection_plans {
            match plan.connection_type {
                FanoutType::Forward { sparsity, polarity } |
                FanoutType::Backward { sparsity, polarity } |
                FanoutType::LongRange { sparsity, polarity } => {
                    self.connect_range(
                        plan.source_range,
                        plan.target_range,
                        plan.sensory,
                        sparsity,
                        polarity,
                        device
                    )?;
                },
                FanoutType::Recurrent { count, polarity } => {
                    self.connect_recurrent_range(
                        plan.source_range.clone(),
                        plan.sensory,
                        count,
                        polarity,
                        device
                    )?;
                }
            }
        }
        Ok(())
    }

    fn create_connection_plans(
        &self
    ) -> Result<Vec<FanoutPlan>, WiringError> {
        let mut plans = Vec::new();
        
        for (idx, layer) in self.layers.iter().enumerate() {
            for conn in &layer.config.connections {
                let source_range = if layer.id == 0 {
                    0..self.input_size
                } else {
                    layer.start_idx..layer.end_idx
                };
                match conn {
                    Fanout::Forward(sparsity, polarity) => {
                        if idx < self.layers.len() - 1 {
                            let next_layer = &self.layers[idx + 1];
                            plans.push(FanoutPlan {
                                _source_id: layer.id,
                                _target_id: next_layer.id,
                                source_range: source_range,
                                target_range: next_layer.start_idx..next_layer.end_idx,
                                connection_type: FanoutType::Forward {
                                    sparsity: *sparsity,
                                    polarity: *polarity,
                                },
                                sensory: layer.id == 0
                            });
                        }
                    },
                    Fanout::Backward(sparsity, polarity) => {
                        if idx > 0 {
                            let prev_layer = &self.layers[idx - 1];
                            plans.push(FanoutPlan {
                                _source_id: layer.id,
                                _target_id: prev_layer.id,
                                source_range: source_range,
                                target_range: prev_layer.start_idx..prev_layer.end_idx,
                                connection_type: FanoutType::Backward {
                                    sparsity: *sparsity,
                                    polarity: *polarity,
                                },
                                sensory: layer.id == 0
                            });
                        }
                    },
                    Fanout::Recurrent(count, polarity) => {
                        plans.push(FanoutPlan {
                            _source_id: layer.id,
                            _target_id: layer.id,
                            source_range: source_range.clone(),
                            target_range: source_range,
                            connection_type: FanoutType::Recurrent {
                                count: *count,
                                polarity: *polarity,
                            },
                            sensory: layer.id == 0
                        });
                    },
                    Fanout::LongRange(target_idx, sparsity, polarity) => {
                        if *target_idx < self.layers.len() {
                            let target_layer = &self.layers[*target_idx];
                            plans.push(FanoutPlan {
                                _source_id: layer.id,
                                _target_id: target_layer.id,
                                source_range: source_range,
                                target_range: target_layer.start_idx..target_layer.end_idx,
                                connection_type: FanoutType::LongRange {
                                    sparsity: *sparsity,
                                    polarity: *polarity,
                                },
                                sensory: layer.id == 0
                            });
                        }
                    },
                    Fanout::None => {}
                }
            }
        }
        Ok(plans)
    }

    fn connect_range(
        &mut self,
        source_range: Range<usize>,
        target_range: Range<usize>,
        sensory: bool,
        sparsity: f64,
        polarity: f64,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let num_source_neurons = source_range.end - source_range.start;
        let num_target_neurons = target_range.end - target_range.start;
        let fanout = std::cmp::max(
            (num_target_neurons as f64 * (1.0 - sparsity)) as usize,
            1
        );
        let mut unreachable_neurons: Vec<usize> = (target_range.start..target_range.end).collect();
    
        for src_offset in 0..num_source_neurons {
            let src = source_range.start + src_offset;
            let dest_indices: Vec<usize> = (0..fanout)
                .map(|_| {
                    let offset = self.rng.gen_range(0..num_target_neurons);
                    let dest = target_range.start + offset;
                    unreachable_neurons.retain(|&x| x != dest);
                    dest
                })
                .collect();
    
            for &dest in &dest_indices {
                let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
                if sensory {
                    self.add_sensory_synapse(src, dest, polarity_value, device)?;
                } else {
                    self.add_synapse(src, dest, polarity_value, device)?;
                }
            }
        }
    
        if !unreachable_neurons.is_empty() {
            let mean_fanin = std::cmp::max(
                (num_source_neurons * fanout) / num_target_neurons,
                1
            );
            
            for &dest in &unreachable_neurons {
                for _ in 0..mean_fanin {
                    let src_offset = self.rng.gen_range(0..num_source_neurons);
                    let src = source_range.start + src_offset;
                    let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
                    if sensory {
                        self.add_sensory_synapse(src, dest, polarity_value, device)?;
                    } else {
                        self.add_synapse(src, dest, polarity_value, device)?;
                    }
                }
            }
        }
    
        Ok(())
    }

    fn connect_recurrent_range(
        &mut self,
        range: Range<usize>,
        sensory: bool,
        count: usize,
        polarity: f64,
        device: &B::Device,
    ) -> Result<(), WiringError> {
        let num_neurons = range.end - range.start;
        
        for _ in 0..count {
            let src_offset = self.rng.gen_range(0..num_neurons);
            let dest_offset = self.rng.gen_range(0..num_neurons);
            
            let src = range.start + src_offset;
            let dest = range.start + dest_offset;
            
            let polarity_value = if self.rng.gen_bool(polarity) { 1 } else { -1 };
            
            if sensory == true {
                self.add_sensory_synapse(src, dest, polarity_value, device)?;
            } else {
                self.add_synapse(src, dest, polarity_value, device)?;
            }
        }
        Ok(())
    }

    fn add_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
        device: &B::Device
    ) -> Result<(), WiringError> {
        if src >= self.total_units {
            return Err(WiringError::InvalidSourceNeuron(src, self.total_units));
        }
        if dest >= self.total_units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.total_units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        let value = Tensor::from_data([[polarity as f32]], device);
        self.adjacency_matrix = self.adjacency_matrix.clone()
            .slice_assign([src..src + 1, dest..dest + 1], value);

        Ok(())
    }

    fn add_sensory_synapse(
        &mut self,
        src: usize,
        dest: usize,
        polarity: i32,
        device: &B::Device
    ) -> Result<(), WiringError> {
        if src >= self.input_size {
            return Err(WiringError::InvalidSensoryNeuron(src, self.input_size));
        }
        if dest >= self.total_units {
            return Err(WiringError::InvalidDestinationNeuron(dest, self.total_units));
        }
        if polarity != -1 && polarity != 1 {
            return Err(WiringError::InvalidPolarity(polarity));
        }

        let value = Tensor::from_data([[polarity as f32]], device);
        self.sensory_adjacency_matrix = self.sensory_adjacency_matrix.clone()
            .slice_assign([src..src + 1, dest..dest + 1], value);

        Ok(())
    }

    pub fn erev_initializer(&self) -> Tensor<B, 2> {
        self.adjacency_matrix.clone()
    }

    pub fn sensory_erev_initializer(&self) -> Tensor<B, 2> {
        self.sensory_adjacency_matrix.clone()
    }

    pub fn adjacency_matrix(&self) -> Tensor<B, 2> {
        self.adjacency_matrix.clone()
    }

    pub fn sensory_adjacency_matrix(&self) -> Tensor<B, 2> {
        self.sensory_adjacency_matrix.clone()
    }
    pub fn draw_graph(
        &self, 
        filename: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        // let width = 800u32;
        // let height = 600u32;
        // let margin = 50.0f64;
        let width = 1200u32;
        let height = 800u32;
        let margin = 50.0f64;

        // let connection_positive = RGBColor(0, 100, 0);  // Dark green
        let connection_positive = RGBColor(27, 94, 32);
        // let connection_negative = RGBColor(139, 0, 0);  // Dark red
        let connection_negative = RGBColor(191, 54, 12);

        let root = BitMapBackend::new(filename, (width, height))
            .into_drawing_area();
        // root.fill(&WHITE)?;
        root.fill(&plotters::style::colors::full_palette::AMBER_300)?;
        // root.fill(&plotters::style::colors::full_palette::BLUE_GREY_900)?;
        // root.fill(&plotters::style::colors::full_palette::BLUEGREY_900)?;

        let mut chart_builder = ChartBuilder::on(&root)
            .margin(margin as i32)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..(width as f64),
                0f64..(height as f64)
            )?;

        chart_builder
            .configure_mesh()
            .disable_mesh()
            .disable_axes()
            .draw()?;

        let total_layers = self.layers.len();
        let layer_width = (width as f64 - 2.0 * margin) / total_layers as f64;
        
        let mut graph = DiGraph::<NodeInfo, f32>::new();
        let mut node_indices = HashMap::new();

        // Layer colors
        let layer_colors = vec![
            &BLUE,      // Sensory layer
            &CYAN,      // First hidden layer
            &MAGENTA,   // Second hidden layer
            &YELLOW,    // Third hidden layer
            &GREEN,     // Fourth hidden layer
            &RED,       // Additional layers...
        ];

        let calculate_pos = |idx: usize, total: usize, layer_x: f64| -> (f64, f64) {
            let vertical_range = height as f64 - 2.0 * margin;
            let base_y = if total == 1 {
                height as f64 / 2.0
            } else {
                margin + (vertical_range * idx as f64) / (total - 1) as f64
            };
            
            let wave_amplitude = layer_width * 0.2;
            let x_offset = if idx % 2 == 0 { wave_amplitude } else { -wave_amplitude };
            
            (layer_x + x_offset, base_y)
        };

        // Add sensory and hidden layer nodes
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_x = margin + layer_idx as f64 * layer_width;
            let layer_size = self.neuron_map[layer.id].len();
            
            for local_idx in 0..layer_size {
                let global_idx = self.neuron_map[layer.id][local_idx];
                let (x, y) = calculate_pos(local_idx, layer_size, layer_x);
                
                let node = NodeInfo {
                    id: global_idx,
                    // id: neuron_idx,
                    layer_id: layer_idx,
                    x,
                    y,
                };
                let idx = graph.add_node(node);
                node_indices.insert((layer_idx, global_idx), idx);
            }
        }

        for node in graph.node_indices() {
            let node_info = &graph[node];
        }

        // Add edges from sensory matrix
        let sensory_matrix = self.sensory_adjacency_matrix.clone();
        for src in 0..self.input_size {
            for dest in 0..self.total_units {
                let weight = sensory_matrix
                    .clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar()
                    .elem::<f32>();
                
                if weight != 0.0 {
                    match self.neuron_map.iter().skip(1).position(|layer| layer.contains(&dest)).map(|pos| pos + 1) {
                        Some(dest_layer) => {
                            let src_idx = *node_indices.get(&(0, src)).expect("Source node not found");
                            let dest_idx = *node_indices.get(&(dest_layer, dest)).expect("Destination node not found");
                            graph.add_edge(src_idx, dest_idx, weight);
                        },
                        None => println!("source {:?} not found in any internal layers", src),
                    }
                }
            }
        }

        // Add edges from adjacency matrix
        let adj_matrix = self.adjacency_matrix.clone();
        for src in 0..self.total_units {
            for dest in 0..self.total_units {
                let weight = adj_matrix
                    .clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar()
                    .elem::<f32>();
                
                if weight != 0.0 {
                    match self.neuron_map.iter().skip(1).position(|layer| layer.contains(&src)).map(|pos| pos + 1) {
                        Some(src_layer) => {
                            match self.neuron_map.iter().skip(1).position(|layer| layer.contains(&dest)).map(|pos| pos + 1) {
                                Some(dest_layer) => {
                                    if let (Some(&src_idx), Some(&dest_idx)) = (
                                        node_indices.get(&(src_layer, src)),
                                        node_indices.get(&(dest_layer, dest))
                                    ) {
                                        graph.add_edge(src_idx, dest_idx, weight);
                                    }
                                },
                                None => println!("dest {:?} not found in any internal layers", dest),
                            }
                        },
                        None => println!("source {:?} not found in any internal layers", src),
                    }
                }
            }
        }

        // Draw edges with curves
        for edge in graph.edge_references() {
            let src = &graph[edge.source()];
            let dest = &graph[edge.target()];
            let weight = edge.weight();
            let color = if *weight > 0.0 { &connection_positive } else { &connection_negative };
        
            let dx = dest.x - src.x;
            let dy = dest.y - src.y;
            let ctrl1 = (
                src.x + dx * 0.33,  // Control point at 1/3 distance
                src.y + dy * 0.1    // Small vertical offset
            );
            let ctrl2 = (
                src.x + dx * 0.66,  // Control point at 2/3 distance
                dest.y - dy * 0.1   // Small vertical offset
            );
        
            let steps = 50;
            let curve_points: Vec<(f64, f64)> = (0..=steps)
                .map(|i| {
                    let t = i as f64 / steps as f64;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;
                    
                    let x = src.x * mt3 + 
                           3.0 * ctrl1.0 * mt2 * t +
                           3.0 * ctrl2.0 * mt * t2 +
                           dest.x * t3;
                    let y = src.y * mt3 +
                           3.0 * ctrl1.1 * mt2 * t +
                           3.0 * ctrl2.1 * mt * t2 +
                           dest.y * t3;
                    
                    (x, y)
                })
                .collect();
        
            // Draw the curve
            chart_builder.draw_series(LineSeries::new(
                curve_points.clone(),
                color.filled().stroke_width(2),
            ))?;
        
            // Calculate arrow head
            let arrow_size = 8.0;
            let last_points = &curve_points[curve_points.len() - 2..];
            let dx = last_points[1].0 - last_points[0].0;
            let dy = last_points[1].1 - last_points[0].1;
            let angle = dy.atan2(dx);
        
            let tip = (
                dest.x - arrow_size * angle.cos(),
                dest.y - arrow_size * angle.sin()
            );
            let left = (
                tip.0 - arrow_size * (angle + std::f64::consts::PI * 0.25).cos(),
                tip.1 - arrow_size * (angle + std::f64::consts::PI * 0.25).sin()
            );
            let right = (
                tip.0 - arrow_size * (angle - std::f64::consts::PI * 0.25).cos(),
                tip.1 - arrow_size * (angle - std::f64::consts::PI * 0.25).sin()
            );
        
            // Draw arrow head
            chart_builder.draw_series(std::iter::once(Polygon::new(
                vec![tip, left, right],
                color.filled(),
            )))?;
        }

        // Draw nodes
        for node_idx in graph.node_indices() {
            let node = &graph[node_idx];
            let color = layer_colors[node.layer_id % layer_colors.len()];

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
        let mut legend_items = Vec::new();
        
        // Add layer colors to legend
        for i in 0..self.layers.len() {
            let label = if i == 0 {
                "Sensory"
            } else if i == self.layers.len()-1 {
                "Output"
            } else {
                "Hidden"
            };
            legend_items.push((label, layer_colors[i]));
        }
        
        // Add connection types to legend
        legend_items.push(("Excitatory", &connection_positive));
        legend_items.push(("Inhibitory", &connection_negative));

        // Draw legend items
        for (i, (label, color)) in legend_items.iter().enumerate() {
            let y = height as f64 - margin - (i as f64 * 20.0);
            
            // Draw legend symbol (circle for layers, line for connections)
            if *label == "Excitatory" || *label == "Inhibitory" {
                chart_builder.draw_series(LineSeries::new(
                    vec![
                        (width as f64 - margin - 60.0, y),
                        (width as f64 - margin - 40.0, y),
                    ],
                    color.stroke_width(2),
                ))?;
            } else {
                chart_builder.draw_series(std::iter::once(Circle::new(
                    (width as f64 - margin - 50.0, y),
                    5,
                    color.filled(),
                )))?;
            }

            // Draw legend label
            chart_builder.draw_series(std::iter::once(Text::new(
                *label,
                (width as f64 - margin - 30.0, y),
                ("sans-serif", 12).into_font(),
            )))?;
        }

        root.present()?;
        Ok(())
    }
}