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
pub struct PhotoSuperpixDatasetItem {
    pub photo_pixels: Vec<u8>,
    pub superpix_pixels: Vec<u8>,
}

#[derive(Debug)]
pub struct PhotoSuperpixDataset {
    photo_files: Vec<String>,
    superpix_files: Vec<String>,
}

impl Dataset<PhotoSuperpixDatasetItem> for PhotoSuperpixDataset {
    fn len(&self) -> usize {
        self.photo_files.len()
    }

    fn get(&self, index: usize) -> Option<PhotoSuperpixDatasetItem> {
        if index >= self.len() {
            return None;
        }

        let photo_image = image::open(&self.photo_files[index]).unwrap().into_rgb8();
        let superpix_image = image::open(&self.superpix_files[index])
            .unwrap()
            .into_rgb8();

        Some(PhotoSuperpixDatasetItem {
            photo_pixels: photo_image.into_raw(),
            superpix_pixels: superpix_image.into_raw(),
        })
    }
}

impl PhotoSuperpixDataset {
    pub fn new(dataset_root: impl AsRef<Path>) -> Self {
        let photo_files = Self::load_files_path(dataset_root.as_ref().join("photo"));
        let superpix_files = Self::load_files_path(dataset_root.as_ref().join("target"));

        Self {
            photo_files,
            superpix_files,
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
pub struct PhotoSuperpixDatasetBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct PhotoSuperpixDatasetBatch<B: Backend> {
    pub photo_images: Tensor<B, 4>,
    pub superpix_images: Tensor<B, 4>,
}

impl<B: Backend> PhotoSuperpixDatasetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, PhotoSuperpixDatasetItem, PhotoSuperpixDatasetBatch<B>>
    for PhotoSuperpixDatasetBatcher<B>
{
    fn batch(
        &self,
        items: Vec<PhotoSuperpixDatasetItem>,
        device: &<B as Backend>::Device,
    ) -> PhotoSuperpixDatasetBatch<B> {
        let (height, width, channel) = (256, 256, 3);
        let (photo_images, superpix_images): (Vec<Tensor<B, 4>>, Vec<Tensor<B, 4>>) = items
            .into_iter()
            .map(|item| {
                let photo = TensorData::from_bytes_vec(
                    item.photo_pixels,
                    [channel, height, width],
                    DType::U8,
                );
                let superpix = TensorData::from_bytes_vec(
                    item.superpix_pixels,
                    [channel, height, width],
                    DType::U8,
                );
                (photo, superpix)
            })
            .map(|(photo, superpix)| {
                let anime: Tensor<B, 4> =
                    Tensor::<B, 3>::from_data(photo, &device).unsqueeze_dim(0);
                let photo: Tensor<B, 4> =
                    Tensor::<B, 3>::from_data(superpix, &device).unsqueeze_dim(0);
                (anime, photo)
            })
            .map(|(anime, photo)| (anime / 255.0, photo / 255.0))
            .collect();

        let photo_images = Tensor::cat(photo_images, 0);
        let superpix_images = Tensor::cat(superpix_images, 0);

        PhotoSuperpixDatasetBatch {
            photo_images,
            superpix_images,
        }
    }
}
