use serde::{Deserialize, Serialize};
use itertools::Itertools;

use crate::neuron::Neuron;

#[derive(Debug, Deserialize, Serialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: i32, weights: i32) -> Layer {
        let n = (0..neurons)
            .collect_vec()
            .iter()
            .map(|_| Neuron::new(weights))
            .collect_vec();
        
        Layer {
            neurons: n,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|n| n.feed_forward(inputs.clone()))
            .collect_vec()
    }
}

