use std::fs;

use itertools::Itertools;
use rand::seq::SliceRandom;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::layer::Layer;

#[derive(Debug, Deserialize, Serialize)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(input_size: i32, output_size: i32, layers: Vec<i32>) -> Network {
        let mut l = Vec::new();
        let mut prev_size = input_size;
        for size in layers {
            l.push(Layer::new(size, prev_size));
            prev_size = size;
        }
        l.push(Layer::new(output_size, prev_size));
        
        Network {
            layers: l,
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string(&self).context("Failed to serialize network data")?;
        fs::write(path, json).context("Failed to write network data to json.")?;

        Ok(())
    }

    pub fn load(path: &str) -> Result<Network> {
        let json = fs::read_to_string(path).context(format!("Failed to read {}.", path))?;
        serde_json::from_str(&json).context("Failed to stringify json.")
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.layers[0].neurons[0].weights.len() {
            panic!("Input size does not match network input size");
        }
        
        self.layers
            .iter()
            .fold(inputs, |acc, layer| layer.feed_forward(acc))
    }

    pub fn back_prop(&mut self, inputs: Vec<f32>, targets: Vec<f32>, learning_rate: f32) {
        let mut activations = Vec::new();
        let mut layer_inputs = inputs.clone();

        for layer in &self.layers {
            layer_inputs = layer.feed_forward(layer_inputs);
            activations.push(layer_inputs.clone());
        }

        let mut errors = activations
            .last()
            .unwrap()
            .iter()
            .zip(targets.iter())
            .map(|(o, t)| t - o)
            .collect_vec();

        let layers_len = self.layers.len();
        for (idx, layer) in self.layers.iter_mut().rev().enumerate() {
            let next_layer = if idx == layers_len - 1 {
                &activations[layers_len - 2]
            } else {
                &activations[layers_len - idx - 2]
            };

            let mut new_errors = Vec::new();
            for (neuron, error) in layer.neurons.iter_mut().zip(errors.iter()) {
                for (weight, input) in neuron.weights.iter_mut().zip(next_layer.iter()) {
                    *weight += error * input * learning_rate;
                }
                neuron.bias += error * learning_rate;
                new_errors.push(*error);
            }
            if idx != layers_len - 1 {
                errors = new_errors;
            }
        }
    }
    
    pub fn train(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, epochs: i32, batch_size: usize, learning_rate: f32) {
        for i in 0..epochs {
            println!("Epoch {}", i + 1);
            let mut data: Vec<_> = inputs.iter().zip(targets.iter()).collect();
            let mut rng = rand::thread_rng();
            data.shuffle(&mut rng);

            for batch in data.chunks(batch_size) {
                let batch_inputs: Vec<Vec<f32>> = batch.iter().map(|(input, _)| (*input).clone()).collect();
                let batch_targets: Vec<Vec<f32>> = batch.iter().map(|(_, target)| (*target).clone()).collect();

                for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                    self.back_prop(input.clone(), target.clone(), learning_rate);
                }
            }
        }
    }
}

