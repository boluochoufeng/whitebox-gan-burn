use std::path::Path;

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, Distribution, module::conv2d, ops::ConvOptions},
};
use image::{ImageResult, RgbImage};

fn box_filter<B: Backend>(x: Tensor<B, 4>, r: usize, device: &B::Device) -> Tensor<B, 4> {
    let ch = x.dims()[1];
    let k = 2 * r + 1;
    let weight = 1.0 / (k.pow(2) as f32);
    let box_kernel =
        Tensor::<B, 4>::ones([ch, 1, k, k], device).slice_fill([.., .., .., ..], weight);
    conv2d(
        x,
        box_kernel,
        None,
        ConvOptions::new([1, 1], [r, r], [1, 1], ch),
    )
}

#[allow(non_snake_case)]
pub fn guided_filter<B: Backend>(
    x: Tensor<B, 4>,
    y: Tensor<B, 4>,
    r: usize,
    eps: f32,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    let N = box_filter::<B>(Tensor::ones([1, 1, h, w], device), r, device);

    let mean_x = box_filter(x.clone(), r, device) / N.clone();
    let mean_y = box_filter(y.clone(), r, device) / N.clone();
    let cov_xy =
        box_filter(x.clone() * y.clone(), r, device) / N.clone() - mean_x.clone() * mean_y.clone();
    let var_x =
        box_filter(x.clone() * x.clone(), r, device) / N.clone() - mean_x.clone() * mean_x.clone();

    let A = cov_xy / (var_x + eps);
    let b = mean_y - A.clone() * mean_x;

    let mean_A = box_filter(A, r, device) / N.clone();
    let mean_b = box_filter(b, r, device) / N;

    mean_A * x + mean_b
}

pub enum ColorShiftMode {
    Normal,
    Uniform,
}

pub fn color_shift<B: Backend>(
    image1: Option<Tensor<B, 4>>,
    image2: Option<Tensor<B, 4>>,
    mode: ColorShiftMode,
    device: &B::Device,
) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
    let (r_weight, g_weight, b_weight) = match mode {
        ColorShiftMode::Normal => (
            Tensor::<B, 1>::random([1], Distribution::Normal(0.299, 0.1), device),
            Tensor::<B, 1>::random([1], Distribution::Normal(0.587, 0.1), device),
            Tensor::<B, 1>::random([1], Distribution::Normal(0.114, 0.1), device),
        ),
        ColorShiftMode::Uniform => (
            Tensor::<B, 1>::random([1], Distribution::Uniform(0.199, 0.399), device),
            Tensor::<B, 1>::random([1], Distribution::Uniform(0.487, 0.687), device),
            Tensor::<B, 1>::random([1], Distribution::Uniform(0.014, 0.214), device),
        ),
    };
    let weights = Tensor::cat(vec![r_weight, g_weight, b_weight], 0).reshape([1, 3, 1, 1]);
    let denorm = weights.clone().sum_dim(1);
    let output1 = image1.map(|img| (img * weights.clone()).sum_dim(1) / denorm.clone());
    let output2 = image2.map(|img| (img * weights).sum_dim(1) / denorm);

    (output1, output2)
}

pub fn save_images<B: Backend, Q: AsRef<Path>>(
    images: Tensor<B, 4>,
    nrow: u32,
    path: Q,
) -> ImageResult<()> {
    let ncol = ((images.dims()[0]) as f32 / nrow as f32).ceil() as u32;
    let width = images.dims()[3] as u32;
    let height = images.dims()[2] as u32;

    let mut imgbuf = RgbImage::new(width * ncol, height * nrow);
    for row in 0..nrow {
        for col in 0..ncol {
            let idx = (row * ncol + col) as usize;
            let image: Tensor<B, 3> = images
                .clone()
                .slice(idx..idx + 1)
                .squeeze_dim(0)
                .swap_dims(0, 1)
                .swap_dims(1, 2)
                .swap_dims(0, 1);
            let image = ((image * 0.5) + 0.5) * 127.5;
            let image: Vec<u8> = image
                .into_data()
                .convert_dtype(DType::U8)
                .iter::<u8>()
                .collect();

            let image = RgbImage::from_vec(width, height, image).unwrap();
            for (x, y, pixel) in image.enumerate_pixels() {
                imgbuf.put_pixel(width * col + x, height * row + y, *pixel);
            }
        }
    }
    imgbuf.save(path)
}

#[cfg(test)]
mod tests {
    use burn::{Tensor, backend::Wgpu, tensor::Distribution};

    use crate::utils::{color_shift, guided_filter};

    type MyBackend = Wgpu<f32, i32>;

    #[test]
    fn test_guided_filter() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let input = Tensor::<MyBackend, 4>::random(
            [1, 3, 256, 256],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = guided_filter(input.clone(), input, 5, 2e-1, &device);
        assert_eq!([1, 3, 256, 256], output.dims());
    }

    #[test]
    fn test_color_shift() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let input = Tensor::<MyBackend, 4>::random(
            [1, 3, 256, 256],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let (input_gray, _) =
            color_shift(Some(input), None, super::ColorShiftMode::Uniform, &device);

        assert!(input_gray.is_some());
        assert_eq!([1, 1, 256, 256], input_gray.unwrap().dims());
    }
}
