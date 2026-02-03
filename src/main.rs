#![recursion_limit = "256"]
use std::time::Instant;

use burn::{
    backend::{Autodiff, Wgpu},
    tensor::backend::AutodiffBackend,
};

use crate::{pretrain::PretrainingConfig, train::TrainingConfig};

mod data;
mod discriminator;
mod generator;
mod pretrain;
mod train;
mod utils;
mod vgg;

fn pretrain<B: AutodiffBackend>(device: &B::Device) {
    let config = PretrainingConfig::new(
        "/home/syx/Code/Rust/whitebox-gan-burn/data/scenery_photo".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/face_photo".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/test".to_owned(),
    )
    .with_batch_size(16)
    .with_total_iter(100);
    pretrain::train::<B>(config, device);
}

fn train<B: AutodiffBackend>(device: &B::Device) {
    let config = TrainingConfig::new(
        "/home/syx/Code/Rust/whitebox-gan-burn/data/scenery_photo".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/face_photo".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/scenery_cartoon/hayao".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/face_cartoon/pa_face".to_owned(),
        "/home/syx/Code/Rust/whitebox-gan-burn/data/test".to_owned(),
    )
    .with_batch_size(16);
    train::train::<B>(config, device);
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    // pretrain::<MyAutodiffBackend>(&device);
    train::<MyAutodiffBackend>(&device);
}
