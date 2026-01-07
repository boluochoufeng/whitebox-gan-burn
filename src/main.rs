use burn::{
    Tensor,
    backend::{Autodiff, Wgpu},
};

use crate::generator::Generator;

mod data;
mod discriminator;
mod generator;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let generator: Generator<MyAutodiffBackend> = generator::GeneratorConfig::new().init(&device);

    let input: Tensor<MyAutodiffBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
    let output = generator.forward(input);
    println!("{}", generator);
    println!("{:?}", output.dims());

    let (discriminator, mut sts) =
        discriminator::DiscriminatorConfig::new().init::<MyAutodiffBackend>(&device);
    let input: Tensor<MyAutodiffBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
    let output = discriminator.forward(input, &mut sts);
    println!("{}", generator);
    println!("{:?}", output.dims());
}
