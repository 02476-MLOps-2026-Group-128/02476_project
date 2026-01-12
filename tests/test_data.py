from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from diabetic_classification import constants
from diabetic_classification.data import DiabetesHealthDataset
from tests import _DATA_DIR, _DATA_SIZE


def _ensure_processed_data(data_dir: Path = _DATA_DIR) -> None:
    processed_dir = data_dir / "processed"
    if processed_dir.exists() and any(processed_dir.iterdir()):
        return
    DiabetesHealthDataset(data_dir).prepare_data()


def test_processed_train_split_is_normalized():
    _ensure_processed_data()

    processed_dir = _DATA_DIR / "processed"
    train_df = pd.read_csv(processed_dir / "train_data.csv")

    numeric_columns = [
        column
        for column in train_df.columns
        if not any(token in column for token in DiabetesHealthDataset.POSSIBLE_TARGET_ATTRIBUTES)
        and not any(token in column for token in DiabetesHealthDataset.CATEGORICAL_ATTRIBUTES)
    ]

    assert numeric_columns, "Expected at least one numerical feature column after preprocessing."

    feature_means = train_df[numeric_columns].mean()
    assert np.allclose(feature_means.values, 0.0, atol=1e-5), "Numerical features are not zero-centered."

    feature_stds = train_df[numeric_columns].std()
    non_constant_mask = feature_stds > 1e-8
    assert np.allclose(
        feature_stds[non_constant_mask], 1.0, atol=1e-2
    ), "Numerical features are not properly scaled, standard deviation should be 1."


def test_dataset_handles_single_target_attribute():
    _ensure_processed_data()
    dataset = DiabetesHealthDataset(_DATA_DIR, target_attributes="diagnosed_diabetes")
    dataset.setup("fit")

    assert dataset.train_dataset is not None, "Train dataset should be initialized."
    example_features, example_target = dataset.train_dataset[0]

    assert example_features.ndim == 1, "The feature dimension should be 1D."
    assert example_target.ndim == 0 or example_target.ndim == 1, "The target dimension should be 0D or 1D."


def test_dataset_handles_multiple_target_attributes():
    _ensure_processed_data()
    target_attributes = ["diabetes_risk_score", "diagnosed_diabetes"]
    dataset = DiabetesHealthDataset(_DATA_DIR, target_attributes=target_attributes)
    dataset.setup("fit")

    assert dataset.train_dataset is not None, "The train dataset should initialized."
    _, example_target = dataset.train_dataset[0]
    assert example_target.shape[-1] == len(
        target_attributes
    ), "The target dimension does not match the number of target attributes."


def test_dataset_respects_split_sizes():
    _ensure_processed_data()
    val_split = 0.15
    dataset = DiabetesHealthDataset(_DATA_DIR, val_split=val_split)
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


def test_dataset_allows_feature_subset():
    _ensure_processed_data()
    feature_attributes = ["age", "bmi", "gender"]
    dataset = DiabetesHealthDataset(
        _DATA_DIR,
        target_attributes="diagnosed_diabetes",
        feature_attributes=feature_attributes,
    )
    dataset.setup("fit")

    assert dataset.train_dataset is not None, "The train dataset should be initialized."

    train_df = dataset._load_split("train_data.csv")
    target_columns = dataset._ensure_target_columns(train_df.columns)
    stratify_series = dataset._build_stratify_series(train_df, dataset.stratification_attributes)
    expected_train_df, _ = train_test_split(
        train_df,
        test_size=dataset.val_split,
        random_state=constants.SEED,
        stratify=stratify_series,
    )

    expected_columns: list[str] = []
    for attribute in feature_attributes:
        expected_columns.extend(dataset._resolve_columns(train_df.columns, attribute))
    expected_columns = [column for column in expected_columns if column not in target_columns]
    seen: set[str] = set()
    expected_columns = [column for column in expected_columns if not (column in seen or seen.add(column))]

    assert dataset.feature_columns == expected_columns

    expected_array = expected_train_df[expected_columns].to_numpy(dtype="float32")
    (
        np.testing.assert_allclose(
            dataset.train_dataset.features.numpy(),
            expected_array,
            rtol=1e-5,
            atol=1e-6,
        ),
        "The feature arrays do not match the expected subset.",
    )


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

    assert len(pd.read_csv(processed_dir / "train_data.csv")) == (1 - constants.TEST_SIZE) * _DATA_SIZE
    assert len(pd.read_csv(processed_dir / "test_data.csv")) == constants.TEST_SIZE * _DATA_SIZE
