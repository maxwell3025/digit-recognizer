#![allow(dead_code)]
#![allow(unused_variables)]
mod image;
mod loader;
mod network;

#[test]
fn gradient(){
    for step_size in [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]{
        let model_a = network::NetworkParameters::new();
        let mut input = image::Image::new();
        input.data[0] = 255;
        let label = 4;
        let grad = model_a.gradient(&input, label);
        let loss_a = model_a.loss(&input, label);
        let norm_squared = grad.layer12.norm_squared() + grad.layer23.norm_squared();
        let model_b = model_a + grad * step_size;
        let loss_b = model_b.loss(&input, label);
        println!("expected loss: {}", step_size * norm_squared);
        println!("actual loss:   {}", loss_b - loss_a);
    }
}
const BETA_1: f64 = 0.9;
const BETA_2: f64 = 0.999;
const EPSILON: f64 = 10e-8;
const ALPHA: f64 = 0.001;
fn main(){
    println!("loading data");
    let mut model = network::NetworkParameters::new();
    let training_images = loader::load_image_set("data/train-images".to_owned()).unwrap();
    let training_labels = loader::load_label_set("data/train-labels".to_owned()).unwrap();
    let test_images = loader::load_image_set("data/test-images".to_owned()).unwrap();
    let test_labels = loader::load_label_set("data/test-labels".to_owned()).unwrap();
    println!("done");

    //SGD
    let mut total_step = 0;
    let mut momentum = network::NetworkParameters::zeroes();
    let mut variance = network::NetworkParameters::zeroes();
    for epoch in 0..10{
        println!("testing model");
        let mut total_loss = 0.0;
        let mut label_count = 0;
        for (index, (image, label)) in training_images.iter().zip(training_labels.iter()).enumerate().filter(|(i, _)| i % 1000 == 0){
            total_loss += model.loss(image, *label);
            label_count += 1;
            print!("\x1B[2K\rimage #{index} tested\n\x1B[1F");
        }
        println!("\x1B[2Ktesting complete");
        println!("avg loss: {}", total_loss / label_count as f64);
        println!("training epoch {} started", epoch + 1);
        for (step, (image, label)) in training_images.iter().zip(training_labels.iter()).enumerate().take(2000){
            total_step += 1;
            //println!("step = {}", step);
            if step % 100 == 0{
                print!("\x1B[2K\x1B[0Gtrained image {}\x1B[1F\n", step);
            }
            let grad = model.gradient(image, *label);
            momentum = momentum.clone() * BETA_1 + grad.clone() * (1.0 - BETA_1);
            variance = variance.clone() * BETA_2 + grad.clone().square() * (1.0 - BETA_2);
            let corrected_momentum = momentum.clone() * (1.0 / (1.0 - BETA_1.powi(total_step)));
            let corrected_variance = variance.clone() * (1.0 / (1.0 - BETA_2.powi(total_step)));
            model = model.clone() + corrected_momentum.clone().divide(&corrected_variance.adam_term(EPSILON))  * (-ALPHA);
        }
        println!("\x1B[2Kepoch training complete");
    }
}
