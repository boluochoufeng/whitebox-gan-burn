use std::time::Instant;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::DefaultRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    data::{PhotoDataset, PhotoDatasetBatcher},
    generator::GeneratorConfig,
    utils::save_images,
};

#[derive(Config, Debug)]
pub struct PretrainingConfig {
    #[config(default = "50000")]
    pub total_iter: usize,
    #[config(default = "32")]
    pub batch_size: usize,
    #[config(default = "4")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.0002")]
    pub lr: f64,
    pub scenery_photo_root: String,
    pub face_photo_root: String,
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
        .num_workers(1)
        .build(PhotoDataset::new(&config.face_photo_root));

    println!("Start training ...");
    let start = Instant::now();

    let mut scenery_iter = scenery_photo_loader.iter();
    let mut face_iter = face_photo_loader.iter();
    for step in 0..config.total_iter {
        let photo = if step % 5 == 0 {
            face_iter.next().unwrap_or_else(|| {
                face_iter = face_photo_loader.iter();
                face_iter.next().unwrap()
            })
        } else {
            scenery_iter.next().unwrap_or_else(|| {
                scenery_iter = face_photo_loader.iter();
                scenery_iter.next().unwrap()
            })
        };

        let output = generator.forward(photo.photo_images.clone());
        let l1_loss = (output.clone() - photo.photo_images).abs().mean();
        let grads = l1_loss.backward();
        let grads = GradientsParams::from_grads(grads, &generator);
        generator = optimizer_g.step(config.lr, generator, grads);

        if (step + 1) % 100 == 0 {
            println!(
                "[Train - Step {}] LossG {:.3}",
                step,
                l1_loss.clone().into_scalar(),
            );

            let _ = save_images(output, 4, &format!("results/pretrain/{step}.jpg"));
            generator
                .clone()
                .save_file(
                    format!("model/pretrain/net_{step}"),
                    &DefaultRecorder::new(),
                )
                .expect("Generator should be saved successfully");
        }
    }

    println!("Time Spend: {:.2}", start.elapsed().as_secs_f32());
}
