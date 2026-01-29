#![recursion_limit = "256"]
use burn::{backend::Autodiff, tensor::backend::AutodiffBackend};

#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(not(feature = "rocm"))]
use burn::backend::Wgpu;

use crate::pretrain::PretrainingConfig;

mod data;
mod discriminator;
mod generator;
mod pretrain;
mod training;
mod utils;
mod vgg;

fn pretrain<B: AutodiffBackend>(device: &B::Device) {
    let config = PretrainingConfig::new(
        "/home/syx/Code/Rust/whitebox-gan-burn/data/scenery_photo".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/face_photo".to_owned(),
    );
    pretrain::train::<B>(config, device);
}

fn main() {
    #[cfg(not(feature = "rocm"))]
    type MyBackend = Wgpu<f32, i32>;
    #[cfg(feature = "rocm")]
    type MyBackend = Rocm<f32, i32>;

    type MyAutodiffBackend = Autodiff<MyBackend>;

    #[cfg(not(feature = "rocm"))]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(feature = "rocm")]
    let device = burn::backend::rocm::RocmDevice::default();

    pretrain::<MyAutodiffBackend>(&device);
}
