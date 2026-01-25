use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    superpix::{
        data::{PhotoSuperpixDataset, PhotoSuperpixDatasetBatcher},
        superpix_net::SuperpixNetConfig,
    },
    utils,
};

#[derive(Config, Debug)]
pub struct SuperpixNetTrainingConfig {
    #[config(default = "10")]
    pub num_epochs: usize,
    #[config(default = "64")]
    pub batch_size: usize,
    #[config(default = "4")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.0001")]
    pub lr: f64,
    pub optimizer: AdamConfig,
}

pub fn train<B: AutodiffBackend>(device: &B::Device) {
    let mut model = SuperpixNetConfig::new().init::<B>(device);

    let optimizer = AdamConfig::new();
    let training_config = SuperpixNetTrainingConfig::new(optimizer);

    let mut optimizer = training_config.optimizer.init();

    let train_batcher = PhotoSuperpixDatasetBatcher::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(train_batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(PhotoSuperpixDataset::new(""));

    let test_batcher = PhotoSuperpixDatasetBatcher::new(device.clone());
    let dataloader_test = DataLoaderBuilder::new(test_batcher.clone())
        .batch_size(16)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(PhotoSuperpixDataset::new(""));

    for epoch in 1..=training_config.num_epochs {
        for (step, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.photo_images);
            let l1_loss = (output - batch.superpix_images).abs().mean();
            println!(
                "[Train - Epoch {} - Step {}] Loss {:.3}",
                epoch,
                step,
                l1_loss.clone().into_scalar(),
            );

            let grads = l1_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(training_config.lr, model, grads);
        }

        let model_valid = model.valid();
        let mut losses = Vec::new();
        for (step, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.photo_images);
            let loss = (output.clone() - batch.superpix_images).abs();
            losses.push(loss.into_scalar());
            let _ =
                utils::save_images(output, 4, format!("result/superpix/{}_{}.png", epoch, step));
        }
        model
            .clone()
            .save_file(
                format!("model/superpix/superpix_{epoch}"),
                &CompactRecorder::new(),
            )
            .expect_err("SuperpixNet should be saved successfully");
    }
}
