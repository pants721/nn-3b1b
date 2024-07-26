use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Neuron {
    pub bias: f32,
    pub weights: Vec<f32>,
}

impl Neuron {
    pub fn new(weights: i32) -> Neuron {
        let mut rng = rand::thread_rng();
        let w = (0..weights)
            .collect_vec()
            .iter()
            .map(|_| rng.gen_range(-1.0..1.0)) // random weights between -1 and 1
            .collect_vec();
        
        Neuron {
            bias: rng.gen_range(-1.0..1.0), // random bias between -1 and 1
            weights: w,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> f32 {
        let sum = self.weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f32>() + self.bias;
        sum
    }
}
