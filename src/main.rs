use std::fs;

use anyhow::Result;

use network::Network;

mod network;
mod layer;
mod neuron;

fn main() -> Result<()> {
    let mut nn = Network::new(2, 1, vec![4]);

    let train_data: Vec<Vec<f32>> = serde_json::from_str(&fs::read_to_string("train_data.json").expect("Unable to read file")).unwrap();
    let target_data: Vec<Vec<f32>> = serde_json::from_str(&fs::read_to_string("target_data.json").expect("Unable to read file")).unwrap();

    nn.train(train_data, target_data, 10, 100, 0.1);
    nn.save("recent.json")?;

    let test_data = [
        vec![1.0 / 10.0, -2.2 / 10.0], 
        vec![3.0 / 10.0, 4.0 / 10.0], 
        vec![11.0 / 10.0, 89.0 / 10.0],
        vec![11111.0 / 10.0, 100000.0 / 10.0]
    ];

    for data in test_data.iter() {
        let result = nn.feed_forward(data.to_vec());
        println!("{:?} + {:?} = {:?}", data[0] * 10.0, data[1] * 10.0, result[0] * 10.0);
    }

    Ok(())
}
