use std::time::Instant;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::DefaultRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    data::{PhotoDataset, PhotoDatasetBatcher, next_or_reset},
    generator::GeneratorConfig,
    utils::save_images,
};

#[derive(Config, Debug)]
pub struct PretrainingConfig {
    #[config(default = "10000")]
    pub total_iter: usize,
    #[config(default = "16")]
    pub batch_size: usize,
    #[config(default = "2")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.0002")]
    pub lr: f64,
    pub scenery_photo_root: String,
    pub face_photo_root: String,
    pub test_photo_root: String,
}

pub fn train<B: AutodiffBackend>(config: PretrainingConfig, device: &B::Device) {
    let mut generator = GeneratorConfig::new().init::<B>(device);
    let mut optimizer_g = AdamConfig::new().init();

    let scenery_photo_batcher = PhotoDatasetBatcher;
    let scenery_photo_loader = DataLoaderBuilder::new(scenery_photo_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.scenery_photo_root));

    let face_photo_batcher = PhotoDatasetBatcher;
    let face_photo_loader = DataLoaderBuilder::new(face_photo_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.face_photo_root));

    let test_photo_batcher = PhotoDatasetBatcher;
    let test_photo_loader = DataLoaderBuilder::new(test_photo_batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(PhotoDataset::new(&config.test_photo_root));

    println!("Start pretraining ...");
    println!("{}", generator);
    let start = Instant::now();

    let mut scenery_iter = scenery_photo_loader.iter();
    let mut face_iter = face_photo_loader.iter();
    for step in 1..=config.total_iter {
        let photo = if step % 5 == 0 {
            next_or_reset(&mut face_iter, || face_photo_loader.iter())
        } else {
            next_or_reset(&mut scenery_iter, || scenery_photo_loader.iter())
        };

        let input = photo.images;
        let output = generator.forward(input.clone());
        let l1_loss = (output.clone() - input).abs().mean();
        let grads = l1_loss.backward();
        let grads = GradientsParams::from_grads(grads, &generator);
        generator = optimizer_g.step(config.lr, generator, grads);

        if step % 100 == 0 {
            println!(
                "[Train - Step {}] LossG {:.3}",
                step,
                l1_loss.clone().into_scalar(),
            );

            let generator_valid = generator.valid();
            for (idx, photo) in test_photo_loader.iter().enumerate() {
                let output = generator_valid.forward(photo.images);
                let _ = save_images(output, 4, &format!("results/pretrain/{step}_{idx}.jpg"));
            }
            generator
                .clone()
                .save_file(
                    format!("model/pretrain/netG_{step}"),
                    &DefaultRecorder::new(),
                )
                .expect("Generator should be saved successfully");
        }
    }

    println!("Time Spend: {:.2}", start.elapsed().as_secs_f32());
}
