#![recursion_limit = "256"]
use std::time::Instant;

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    data::{PhotoDataset, PhotoDatasetBatch, PhotoDatasetBatcher, PhotoDatasetItem},
    pretrain::PretrainingConfig,
    utils::{save_images, simple_superpix_batch},
};

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
    .with_batch_size(16);
    pretrain::train::<B>(config, device);
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    // pretrain::<MyAutodiffBackend>(&device);

    let batcher = PhotoDatasetBatcher;
    let data_loader =
        DataLoaderBuilder::<MyBackend, PhotoDatasetItem, PhotoDatasetBatch<MyBackend>>::new(
            batcher,
        )
        .batch_size(16)
        .build(PhotoDataset::new(
            "/home/syx/Code/Rust/whitebox-gan-burn/data/scenery_photo",
        ));

    let start = Instant::now();
    for (idx, batch) in data_loader.iter().enumerate() {
        let output = simple_superpix_batch(batch.images, 200, 10.0, 10, 1.0, None, None, &device);
        let _ = save_images(output, 4, format!("./results/superpixel/{idx}.jpg"));
    }
    println!("{:.2}s", start.elapsed().as_secs_f32());
}
