import os
from pathlib import Path
import shutil

import kagglehub
import typer
from torch.utils.data import Dataset

import config


class DiabetesHealthDataset(Dataset):
    """"""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        # Check if data dir exists or is empty
        if not self.data_path.exists() or not any(self.data_path.iterdir()):
            self.prepare_data(data_path)

    def prepare_data(self, data_path: Path) -> None:
        """Download the data if it doesn't exist."""
        cache_path = kagglehub.dataset_download(config.DATASET_NAME)

        dest_path = Path("data/raw")
        os.makedirs(dest_path, exist_ok=True)

        # Move all files from cache to your local folder
        for file in os.listdir(cache_path):
            shutil.move(os.path.join(cache_path, file), os.path.join(dest_path, file))

        self.preprocess_data(dest_path, data_path)

    def preprocess_data(self, raw_data_dir: Path, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        pass  # Implement preprocessing logic here

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = DiabetesHealthDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
