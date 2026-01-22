use crate::generator::Generator;
use burn::{
    Tensor,
    backend::{Autodiff, Wgpu},
};

mod data;
mod discriminator;
mod generator;
mod superpix;
mod training;
mod utils;
mod vgg;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let generator: Generator<MyAutodiffBackend> = generator::GeneratorConfig::new().init(&device);

    let input: Tensor<MyAutodiffBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
    let output = generator.forward(input);
    println!("{}", generator);
    println!("{:?}", output.dims());

    let mut discriminator =
        discriminator::DiscriminatorConfig::new().init::<MyAutodiffBackend>(&device);
    let input: Tensor<MyAutodiffBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
    let output = discriminator.forward(input);
    println!("{}", generator);
    println!("{:?}", output.dims());
}
