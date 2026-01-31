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
pub struct AnimePhoteDatasetItem {
    pub anime_pixels: Vec<u8>,
    pub photo_pixels: Vec<u8>,
}

#[derive(Debug)]
pub struct AnimePhotoDataset {
    anime_files: Vec<String>,
    photo_files: Vec<String>,
}

impl Dataset<AnimePhoteDatasetItem> for AnimePhotoDataset {
    fn len(&self) -> usize {
        self.anime_files.len().max(self.photo_files.len())
    }

    fn get(&self, index: usize) -> Option<AnimePhoteDatasetItem> {
        if index >= self.len() {
            return None;
        }

        let anime_file_index = if index >= self.anime_files.len() {
            index % self.anime_files.len()
        } else {
            index
        };
        let anime_image = image::open(&self.anime_files[anime_file_index])
            .unwrap()
            .into_rgb8();

        let photo_file_index = if index >= self.photo_files.len() {
            index % self.photo_files.len()
        } else {
            index
        };
        let photo_image = image::open(&self.photo_files[photo_file_index])
            .unwrap()
            .into_rgb8();

        Some(AnimePhoteDatasetItem {
            anime_pixels: anime_image.into_raw(),
            photo_pixels: photo_image.into_raw(),
        })
    }
}

impl AnimePhotoDataset {
    pub fn new(dataset_root: impl AsRef<Path>) -> Self {
        let anime_files = load_files_path(dataset_root.as_ref().join("anime"));
        let photo_files = load_files_path(dataset_root.as_ref().join("photo"));

        Self {
            anime_files,
            photo_files,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnimePhotoDatasetBatcher;
#[derive(Clone, Debug)]
pub struct AnimePhotoDatasetBatch<B: Backend> {
    pub anime_images: Tensor<B, 4>,
    pub photo_images: Tensor<B, 4>,
}

impl<B: Backend> Batcher<B, AnimePhoteDatasetItem, AnimePhotoDatasetBatch<B>>
    for AnimePhotoDatasetBatcher
{
    fn batch(
        &self,
        items: Vec<AnimePhoteDatasetItem>,
        device: &<B as Backend>::Device,
    ) -> AnimePhotoDatasetBatch<B> {
        let (height, width, channel) = (256, 256, 3);
        let (anime_images, photo_images): (Vec<Tensor<B, 4>>, Vec<Tensor<B, 4>>) = items
            .into_iter()
            .map(|item| {
                let anime = TensorData::from_bytes_vec(
                    item.anime_pixels,
                    [width, height, channel],
                    DType::U8,
                );
                let photo = TensorData::from_bytes_vec(
                    item.photo_pixels,
                    [width, height, channel],
                    DType::U8,
                );
                (anime, photo)
            })
            .map(|(anime, photo)| {
                let anime: Tensor<B, 4> = Tensor::<B, 3>::from_data(anime, &device)
                    .permute([2, 1, 0])
                    .unsqueeze_dim(0);
                let photo: Tensor<B, 4> = Tensor::<B, 3>::from_data(photo, &device)
                    .permute([2, 1, 0])
                    .unsqueeze_dim(0);
                (anime, photo)
            })
            .map(|(anime, photo)| (anime / 255.0, photo / 255.0))
            .collect();

        let anime_images = Tensor::cat(anime_images, 0);
        let photo_images = Tensor::cat(photo_images, 0);

        AnimePhotoDatasetBatch {
            anime_images,
            photo_images,
        }
    }
}

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn check_length() {
        let dataset =
            AnimePhotoDataset::new("/home/syx/Code/Rust/whitebox-gan-burn/data/train".to_owned());
        if let Some(item) = dataset.get(2000) {
            println!("{}", item.anime_pixels.len());
            println!("{}", item.photo_pixels.len());
        }

        assert_eq!(6656, dataset.len());
    }
}
