use std::time::Instant;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    record::DefaultRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    data::{PhotoDataset, PhotoDatasetBatcher, next_or_reset},
    discriminator::DiscriminatorConfig,
    generator::{Generator, GeneratorConfig},
    utils::{color_shift, guided_filter, save_images},
    vgg::{VGG19, load_vgg_model},
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = "50000")]
    pub total_iter: usize,
    #[config(default = "16")]
    pub batch_size: usize,
    #[config(default = "2")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.0002")]
    pub g_lr: f64,
    #[config(default = "0.0002")]
    pub d_lr: f64,
    pub scenery_photo_root: String,
    pub face_photo_root: String,
    pub scenery_cartoon_root: String,
    pub face_cartoon_root: String,
    pub test_photo_root: String,
}

fn load_pretrain_generator<B: Backend>(
    generator: Generator<B>,
    load_step: usize,
    device: &B::Device,
) -> Generator<B> {
    let path = format!("./result/pretrain/netG_{load_step}.mpk");
    generator
        .load_file(&path, &DefaultRecorder::new(), device)
        .expect(&format!("No exist model record: {path}"))
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: &B::Device) {
    // 数据集
    let scenery_photo_batcher = PhotoDatasetBatcher;
    let scenery_photo_loader = DataLoaderBuilder::new(scenery_photo_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.scenery_photo_root));

    let scenery_cartoon_batcher = PhotoDatasetBatcher;
    let scenery_cartoon_loader = DataLoaderBuilder::new(scenery_cartoon_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.scenery_cartoon_root));

    let face_photo_batcher = PhotoDatasetBatcher;
    let face_photo_loader = DataLoaderBuilder::new(face_photo_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.face_photo_root));

    let face_cartoon_batcher = PhotoDatasetBatcher;
    let face_cartoon_loader = DataLoaderBuilder::new(face_cartoon_batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PhotoDataset::new(&config.face_cartoon_root));

    let test_photo_batcher = PhotoDatasetBatcher;
    let test_photo_loader = DataLoaderBuilder::new(test_photo_batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(PhotoDataset::new(&config.test_photo_root));

    // 模型定义
    let generator = GeneratorConfig::new().init::<B>(device);
    let mut generator = load_pretrain_generator(generator, 9999, device);
    let mut optimizer_g = AdamConfig::new().init();

    let mut discriminator_gray = DiscriminatorConfig::new()
        .with_in_channel(1)
        .init::<B>(device);
    let mut optimizer_gray = AdamConfig::new().init();
    let mut discriminator_blur = DiscriminatorConfig::new().init::<B>(device);
    let mut optimizer_blur = AdamConfig::new().init();

    let vgg = load_vgg_model::<B>("./", device).expect("VGG model weights file no exist");
    println!("Start pretraining ...");
    println!("{}", generator);
    let start = Instant::now();

    let mut scenery_photo_iter = scenery_photo_loader.iter();
    let mut scenery_cartoon_iter = scenery_cartoon_loader.iter();
    let mut face_photo_iter = face_photo_loader.iter();
    let mut face_cartoon_iter = face_cartoon_loader.iter();
    for step in 0..config.total_iter {
        let (photo, cartoon) = if step % 5 == 0 {
            (
                next_or_reset(&mut face_photo_iter, || face_photo_loader.iter()),
                next_or_reset(&mut face_cartoon_iter, || face_cartoon_loader.iter()),
            )
        } else {
            (
                next_or_reset(&mut scenery_photo_iter, || scenery_photo_loader.iter()),
                next_or_reset(&mut scenery_cartoon_iter, || scenery_cartoon_loader.iter()),
            )
        };

        let input_photo = photo.images;
        let input_cartoon = cartoon.images;
        // 训练生成器
        let mut output = generator.forward(input_photo.clone());
        output = guided_filter(input_photo.clone(), output, 1, 1e-2, device);

        let blur_fake = guided_filter(output.clone(), output.clone(), 5, 2e-1, device);
        let blur_cartoon = guided_filter(
            input_cartoon.clone(),
            input_cartoon.clone(),
            5,
            2e-1,
            device,
        );
        let blur_fake_pred = discriminator_blur.forward(blur_fake);
        let g_loss_blur = (blur_fake_pred - 1).square().mean();

        let (gray_fake, gray_cartoon) = color_shift(
            Some(output.clone()),
            Some(input_cartoon.clone()),
            crate::utils::ColorShiftMode::Normal,
            device,
        );
        let (gray_fake, gray_cartoon) = (gray_fake.unwrap(), gray_cartoon.unwrap());
        let gray_fake_pred = discriminator_gray.forward(gray_fake);
        let g_loss_gray = (gray_fake_pred - 1).square().mean();

        let vgg_photo = vgg.forward(input_photo.clone());
        let vgg_output = vgg.forward(output.clone());
        // let vgg_superpixel = vgg.forward(x);
        //

        let [_, c, h, w] = vgg_photo.dims();
        let photo_loss = (vgg_photo - vgg_output).abs().mean() / (c as i32 * h as i32 * w as i32);

        // let grads = l1_loss.backward();
        // let grads = GradientsParams::from_grads(grads, &generator);
        // generator = optimizer_g.step(config.g_lr, generator, grads);

        if (step + 1) % 100 == 0 {
            // println!(
            //     "[Train - Step {}] LossG {:.3}",
            //     step,
            //     l1_loss.clone().into_scalar(),
            // );

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
