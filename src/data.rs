use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::{
    Tensor,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{DType, TensorData},
};

#[derive(Debug, Clone)]
pub struct PhotoDatasetItem {
    pub pixels: Vec<u8>,
}

#[derive(Debug)]
pub struct PhotoDataset {
    photo_files: Vec<String>,
}

impl Dataset<PhotoDatasetItem> for PhotoDataset {
    fn len(&self) -> usize {
        self.photo_files.len()
    }

    fn get(&self, index: usize) -> Option<PhotoDatasetItem> {
        if index >= self.len() {
            return None;
        }

        let img = match image::open(&self.photo_files[index]) {
            Ok(img) => img.into_rgb8(),
            Err(_) => return None,
        };

        Some(PhotoDatasetItem {
            pixels: img.into_raw(),
        })
    }
}

impl PhotoDataset {
    pub fn new(dataset_root: impl AsRef<Path>) -> Self {
        let photo_files = load_files_path(dataset_root.as_ref().to_path_buf());
        Self { photo_files }
    }
}

#[derive(Clone, Debug)]
pub struct PhotoDatasetBatcher;

#[derive(Clone, Debug)]
pub struct PhotoDatasetBatch<B: Backend> {
    pub images: Tensor<B, 4>,
}

impl<B: Backend> Batcher<B, PhotoDatasetItem, PhotoDatasetBatch<B>> for PhotoDatasetBatcher {
    fn batch(
        &self,
        items: Vec<PhotoDatasetItem>,
        device: &<B as Backend>::Device,
    ) -> PhotoDatasetBatch<B> {
        let (height, width, channel) = (256, 256, 3);
        let photo_images = items
            .into_iter()
            .map(|item| {
                TensorData::from_bytes_vec(item.pixels, [width, height, channel], DType::U8)
            })
            .map(|photo| {
                let photo: Tensor<B, 4> = Tensor::<B, 3>::from_data(photo, &device)
                    .permute([2, 1, 0])
                    .unsqueeze_dim(0);
                photo
            })
            .map(|photo| photo / 127.5 - 1.0)
            .collect();

        let photo_images = Tensor::cat(photo_images, 0);

        PhotoDatasetBatch {
            images: photo_images,
        }
    }
}

fn load_files_path(dir_path: PathBuf) -> Vec<String> {
    fs::read_dir(&dir_path)
        .expect(&format!("Dataset folder {:?} should exist", dir_path))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|path| path.is_file())
        .map(|path| path.to_str().unwrap().to_owned())
        .collect()
}

pub fn next_or_reset<I, R>(iter: &mut I, reset: R) -> I::Item
where
    I: Iterator,
    R: FnOnce() -> I,
{
    iter.next().unwrap_or_else(|| {
        *iter = reset();
        iter.next().expect("Dotaloader shouldn't be empty.")
    })
}
