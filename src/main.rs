use std::fs;

use itertools::Itertools;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Network {
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

    pub fn save(&self, path: &str) {
        let json = serde_json::to_string(&self).unwrap();
        fs::write(path, json).unwrap();
    }

    pub fn load(path: &str) -> Network {
        let json = fs::read_to_string(path).unwrap();
        serde_json::from_str(&json).unwrap()
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

#[derive(Debug, Deserialize, Serialize)]
struct Layer {
    neurons: Vec<Neuron>,    
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

#[derive(Debug, Deserialize, Serialize)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn new(weights: i32) -> Neuron {
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

    fn feed_forward(&self, inputs: Vec<f32>) -> f32 {
        let sum = self.weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f32>() + self.bias;
        sum
    }
}

fn main() {
    let mut nn = Network::new(2, 1, vec![8]);

    let train_data = serde_json::from_str(&fs::read_to_string("train_data.json").expect("Unable to read file")).unwrap();
    let target_data = serde_json::from_str(&fs::read_to_string("target_data.json").expect("Unable to read file")).unwrap();

    nn.train(train_data, target_data, 10, 100, 0.1);
    nn.save("recent.json");

    let test_data = [
        vec![1.0 / 10.0, 2.2 / 10.0], 
        vec![3.0 / 10.0, 4.0 / 10.0], 
        vec![11.0 / 10.0, 89.0 / 10.0]
    ];
    
    for data in test_data.iter() {
        let result = nn.feed_forward(data.to_vec());
        println!("{:?} + {:?} = {:?}", data[0] * 10.0, data[1] * 10.0, result[0] * 10.0);
    }
}
