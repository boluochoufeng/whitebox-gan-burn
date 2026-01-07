use std::{fs, path::PathBuf};

use burn::data::dataset::{Dataset, DatasetIterator};

#[derive(Debug, Clone)]
pub struct AnimePhoteDatasetItem {
    pub anime_pixels: Vec<u8>,
    pub photo_pixels: Vec<u8>,
}

#[derive(Debug)]
pub struct AnimePhoteDataset {
    anime_files: Vec<String>,
    photo_files: Vec<String>,
}

impl Dataset<AnimePhoteDatasetItem> for AnimePhoteDataset {
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
            .into_luma8();

        let photo_file_index = if index >= self.photo_files.len() {
            index % self.photo_files.len()
        } else {
            index
        };
        let photo_image = image::open(self.photo_files[photo_file_index].clone())
            .unwrap()
            .into_luma8();

        Some(AnimePhoteDatasetItem {
            anime_pixels: anime_image.into_raw(),
            photo_pixels: photo_image.into_raw(),
        })
    }
}

impl AnimePhoteDataset {
    pub fn new(dataset_root: &str) -> Self {
        let entries = fs::read_dir(dataset_root).expect("Dataset folder should exist");
        todo!()
    }

    fn load_files_path(path: PathBuf) -> Vec<String> {
        fs::read_dir(&path)
            .expect(&format!("Dataset folder {:?} should exist", path))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|path| path.is_file())
            .map(|path| path.to_str().unwrap().to_owned())
            .collect()
    }
}
