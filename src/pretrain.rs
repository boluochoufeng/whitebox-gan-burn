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
    data::{AnimePhotoDataset, AnimePhotoDatasetBatcher},
    generator::GeneratorConfig,
    utils,
};

#[derive(Config, Debug)]
pub struct PretrainingConfig {
    #[config(default = "4")]
    pub num_epochs: usize,
    #[config(default = "32")]
    pub batch_size: usize,
    #[config(default = "4")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.0002")]
    pub lr: f64,
    pub train_data_root: String,
    pub test_data_root: String,
}

pub fn train<B: AutodiffBackend>(config: PretrainingConfig, device: &B::Device) {
    let mut generator = GeneratorConfig::new().init::<B>(device);
    let mut optimizer_g = AdamConfig::new().init();

    let train_batcher = AnimePhotoDatasetBatcher;
    let dataloader_train = DataLoaderBuilder::new(train_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(AnimePhotoDataset::new(&config.train_data_root));

    let test_batcher = AnimePhotoDatasetBatcher;
    let dataloader_test = DataLoaderBuilder::new(test_batcher.clone())
        .batch_size(16)
        .num_workers(config.num_workers)
        .build(AnimePhotoDataset::new(&config.test_data_root));

    println!("Start training ...");
    let start = Instant::now();
    for epoch in 0..config.num_epochs {
        for (step, batch) in dataloader_train.iter().enumerate() {
            let fake_img = generator.forward(batch.photo_images.clone());
            let l1_loss = (fake_img.clone() - batch.photo_images).abs().mean();
            let loss_g = l1_loss;
            let grads = loss_g.backward();
            let grads = GradientsParams::from_grads(grads, &generator);
            generator = optimizer_g.step(config.lr, generator, grads);

            if step % 100 == 0 {
                println!(
                    "[Train - Epoch {} - Step {}] LossG {:.3}",
                    epoch,
                    step,
                    loss_g.clone().into_scalar(),
                );
            }
        }

        let model_valid = generator.valid();
        for (step, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.photo_images);
            let _ =
                utils::save_images(output, 4, format!("result/superpix/{}_{}.png", epoch, step));
            if step == 10 {
                break;
            }
        }
        generator
            .clone()
            .save_file(
                format!("model/pretrain/superpix_{epoch}"),
                &DefaultRecorder::new(),
            )
            .expect("SuperpixNet should be saved successfully");
    }
    println!("Time Spend: {:.2}", start.elapsed().as_secs_f32());
}
