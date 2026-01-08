use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::{
    Tensor,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, DatasetIterator},
    },
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
        let anime_image = image::open(self.anime_files[anime_file_index].clone())
            .unwrap()
            .into_rgb8();

        let photo_file_index = if index >= self.photo_files.len() {
            index % self.photo_files.len()
        } else {
            index
        };
        let photo_image = image::open(self.photo_files[photo_file_index].clone())
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
        let anime_files = Self::load_files_path(dataset_root.as_ref().join("anime"));
        let photo_files = Self::load_files_path(dataset_root.as_ref().join("photo"));

        Self {
            anime_files,
            photo_files,
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
}

#[derive(Clone, Debug)]
pub struct AnimePhotoDatasetBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct AnimePhotoDatasetBatch<B: Backend> {
    pub anime_images: Tensor<B, 4>,
    pub photo_images: Tensor<B, 4>,
}

impl<B: Backend> AnimePhotoDatasetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, AnimePhoteDatasetItem, AnimePhotoDatasetBatch<B>>
    for AnimePhotoDatasetBatcher<B>
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
                    [channel, height, width],
                    DType::U8,
                );
                let photo = TensorData::from_bytes_vec(
                    item.photo_pixels,
                    [channel, height, width],
                    DType::U8,
                );
                (anime, photo)
            })
            .map(|(anime, photo)| {
                let anime: Tensor<B, 4> =
                    Tensor::<B, 3>::from_data(anime, &device).unsqueeze_dim(0);
                let photo: Tensor<B, 4> =
                    Tensor::<B, 3>::from_data(photo, &device).unsqueeze_dim(0);
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
