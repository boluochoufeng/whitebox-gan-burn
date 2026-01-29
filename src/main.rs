#![recursion_limit = "256"]
use burn::{backend::Autodiff, tensor::backend::AutodiffBackend};

#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(not(feature = "rocm"))]
use burn::backend::Wgpu;

mod data;
mod discriminator;
mod generator;
mod pretrain;
mod training;
mod utils;
mod vgg;

// fn pretrain<B: AutodiffBackend>(device: &B::Device) {
//     let train_data_root = "/home/syx/Code/Rust/whitebox-gan-burn/data/superpix2/train/".to_string();
//     let test_data_root = "/home/syx/Code/Rust/whitebox-gan-burn/data/superpix2/test/".to_string();
//     todo!()
// }

// fn main() {
//     #[cfg(not(feature = "rocm"))]
//     type MyBackend = Wgpu<f32, i32>;
//     #[cfg(feature = "rocm")]
//     type MyBackend = Rocm<f32, i32>;

//     type MyAutodiffBackend = Autodiff<MyBackend>;

//     #[cfg(not(feature = "rocm"))]
//     let device = burn::backend::wgpu::WgpuDevice::default();
//     #[cfg(feature = "rocm")]
//     let device = burn::backend::rocm::RocmDevice::default();

//     todo!()
// }

use image::{Rgb, RgbImage};
use ndarray::Array2;
use scirs2_vision::segmentation::slic;
use std::{collections::HashMap, error::Error, fs, path::Path, time::Instant};

fn main() {
}
