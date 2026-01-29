use std::{collections::HashMap, error::Error, path::Path};

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, Distribution, module::conv2d, ops::ConvOptions},
};
use image::{ImageResult, Rgb, RgbImage};
use ndarray::Array2;
use scirs2_vision::slic;

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
                .permute([2, 1, 0]);
            let image = ((image * 0.5) + 0.5) * 255;
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

fn label2rgb_avg(
    labels: &Array2<i32>, // 改为 Array2<i32> 引用
    image: &RgbImage,
    bg_label: Option<i32>,
    bg_color: Option<[u8; 3]>,
) -> RgbImage {
    let (width, height) = (labels.shape()[0], labels.shape()[1]);
    let mut output = RgbImage::new(width as u32, height as u32);

    // 第一次遍历：收集每个标签的颜色总和和像素计数
    let mut label_stats: HashMap<i32, (usize, [usize; 3])> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]];

            // 跳过背景标签（如果指定了背景标签且当前标签是背景）
            if let Some(bg) = bg_label {
                if label == bg {
                    continue;
                }
            }

            let pixel = image.get_pixel(x as u32, y as u32);
            let entry = label_stats.entry(label).or_insert((0, [0, 0, 0]));

            entry.0 += 1;
            entry.1[0] += pixel[0] as usize;
            entry.1[1] += pixel[1] as usize;
            entry.1[2] += pixel[2] as usize;
        }
    }

    // 计算每个标签的平均颜色
    let mut avg_colors: HashMap<i32, [u8; 3]> = HashMap::new();
    for (&label, &(count, sum)) in &label_stats {
        if count > 0 {
            let avg_color = [
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
            ];
            avg_colors.insert(label, avg_color);
        }
    }

    // 第二次遍历：设置输出颜色
    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]];

            // 处理背景标签
            if let Some(bg) = bg_label {
                if label == bg {
                    if let Some(color) = bg_color {
                        output.put_pixel(x as u32, y as u32, Rgb(color));
                    }
                    continue;
                }
            }

            // 设置标签的平均颜色
            if let Some(&avg_color) = avg_colors.get(&label) {
                output.put_pixel(x as u32, y as u32, Rgb(avg_color));
            }
        }
    }

    output
}

pub fn process_one(path: impl AsRef<Path>, save_path: String) -> Result<(), Box<dyn Error>> {
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    let (n_segments, compactness) = (200, 10.0);
    let labels = slic(&img, n_segments, compactness, 10, 1.0)?;
    let labels = labels.mapv(|x| x as i32);
    let result = label2rgb_avg(&labels, &rgb_img, None, None);
    result.save(format!("result/{}", save_path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, time::Instant};

    use burn::{Tensor, backend::Wgpu, tensor::Distribution};

    use crate::utils::{color_shift, guided_filter, process_one};

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

    #[test]
    fn test_superpix() {
        let dir_path = "/home/syx/Code/Rust/whitebox-gan-burn/data/superpix2/test/photo";
        let files: Vec<String> = fs::read_dir(dir_path)
            .expect(&format!("Dataset folder {:?} should exist", dir_path))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|path| path.is_file())
            .map(|path| path.to_str().unwrap().to_owned())
            .collect();

        let start = Instant::now();
        for (i, file_path) in files.iter().enumerate() {
            let _ = process_one(file_path, format!("{i}.jpg"));
        }
        println!("{}s", start.elapsed().as_secs_f32() / files.len() as f32);
    }
}
