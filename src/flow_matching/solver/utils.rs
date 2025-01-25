// src/flow_matching/solver/utils.rs

use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::ElementConversion;

/// Finds the nearest times in t_discretization for each time in time_grid
pub fn get_nearest_times<B: Backend>(
    time_grid: Tensor<B, 1>,
    t_discretization: Tensor<B, 1>
) -> Tensor<B, 1> {
    let time_grid = time_grid.unsqueeze_dim(1);
    let t_disc = t_discretization.clone().unsqueeze_dim(0);
    
    // Compute distances
    let distances = (time_grid.clone() - t_disc).abs();
    
    // Get indices of minimum distances
    let nearest_indices = distances.argmin(1);
    
    // Index into t_discretization using nearest indices
    let mut result = Tensor::zeros_like(&time_grid);
    for i in 0..nearest_indices.dims()[0] {
        let idx = nearest_indices.clone().slice([i..i+1]);
        let val = t_discretization.clone().slice([idx.clone().into_scalar().elem::<i64>() as usize..idx.into_scalar().elem::<i64>() as usize + 1]);
        result = result.slice_assign([i..i+1], val);
    }
    
    result.squeeze_dims(&[1]) 
}