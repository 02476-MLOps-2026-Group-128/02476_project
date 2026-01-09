from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from diabetic_classification import constants
from diabetic_classification.data import DiabetesHealthDataset


def _ensure_processed_data(data_dir: Path) -> None:
	processed_dir = data_dir / "processed"
	if processed_dir.exists() and any(processed_dir.iterdir()):
		return
	DiabetesHealthDataset(data_dir).prepare_data()


def test_processed_train_split_is_normalized():
	data_dir = Path("data")
	_ensure_processed_data(data_dir)

	processed_dir = data_dir / "processed"
	train_df = pd.read_csv(processed_dir / "train_data.csv")

	numeric_columns = [
		column
		for column in train_df.columns
		if not any(token in column for token in DiabetesHealthDataset.POSSIBLE_TARGET_ATTRIBUTES)
		and not any(token in column for token in DiabetesHealthDataset.CATEGORICAL_ATTRIBUTES)
	]

	assert numeric_columns, "Expected at least one numerical feature column after preprocessing."

	feature_means = train_df[numeric_columns].mean()
	assert np.allclose(feature_means.values, 0.0, atol=1e-5)

	feature_stds = train_df[numeric_columns].std()
	non_constant_mask = feature_stds > 1e-8
	assert np.allclose(feature_stds[non_constant_mask], 1.0, atol=1e-2)


def test_dataset_handles_single_target_attribute():
	data_dir = Path("data")
	_ensure_processed_data(data_dir)
	dataset = DiabetesHealthDataset(data_dir, target_attributes="diagnosed_diabetes")
	dataset.setup("fit")

	assert dataset.train_dataset is not None
	example_features, example_target = dataset.train_dataset[0]

	assert example_features.ndim == 1
	assert example_target.ndim == 0 or example_target.ndim == 1


def test_dataset_handles_multiple_target_attributes():
	data_dir = Path("data")
	_ensure_processed_data(data_dir)
	target_attributes = ["diabetes_risk_score", "diagnosed_diabetes"]
	dataset = DiabetesHealthDataset(data_dir, target_attributes=target_attributes)
	dataset.setup("fit")

	assert dataset.train_dataset is not None
	_, example_target = dataset.train_dataset[0]
	assert example_target.shape[-1] == len(target_attributes)


def test_dataset_respects_split_sizes():
	data_dir = Path("data")
	_ensure_processed_data(data_dir)
	val_split = 0.15
	dataset = DiabetesHealthDataset(data_dir, val_split=val_split)
	dataset.setup("fit")

	train_df = dataset._load_split("train_data.csv")
	stratify_series = dataset._build_stratify_series(train_df, dataset.stratification_attributes)

	expected_train_df, expected_val_df = train_test_split(
		train_df,
		test_size=val_split,
		random_state=constants.SEED,
		stratify=stratify_series,
	)

	assert dataset.train_dataset is not None
	assert dataset.val_dataset is not None
	assert len(dataset.train_dataset) == len(expected_train_df)
	assert len(dataset.val_dataset) == len(expected_val_df)


@pytest.mark.download
def test_prepare_data_downloads_and_processes(tmp_path: Path):
	data_dir = tmp_path / "data"
	data_dir.mkdir()

	dataset = DiabetesHealthDataset(data_dir)
	dataset.prepare_data()

	processed_dir = data_dir / "processed"
	assert (processed_dir / "train_data.csv").exists()
	assert (processed_dir / "test_data.csv").exists()
	assert (processed_dir / "standardization_params.csv").exists()

