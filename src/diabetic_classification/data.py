import os
import shutil
from pathlib import Path
from typing import Optional, Sequence, Tuple

import kagglehub
import pandas as pd
import torch
import typer
from loguru import logger
from pyarrow import csv
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


from diabetic_classification import constants


class DiabetesTabularDataset(Dataset):
    """Torch dataset that wraps tabular diabetes data."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_attributes: Sequence[str],
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize tensors for features and labels.

        Args:
            data: Preprocessed pandas DataFrame containing one data split.
            target_attributes: Attribute names that should be treated as the supervised target.
            feature_columns: Optional explicit feature column selection to retain.
        """
        if not target_attributes:
            raise ValueError("target_attributes must contain at least one column name.")

        missing_columns = [col for col in target_attributes if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Target attributes {missing_columns} not present in the provided DataFrame.")

        if feature_columns is not None:
            if not feature_columns:
                raise ValueError("feature_columns must contain at least one column when provided.")
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Feature columns {missing_features} not present in the provided DataFrame.")
            feature_frame = data[list(feature_columns)]
        else:
            feature_frame = data.drop(columns=list(target_attributes))
            if feature_frame.empty:
                raise ValueError("No feature columns remain after dropping targets.")

        features = feature_frame.to_numpy(dtype="float32")
        target = data[list(target_attributes)].to_numpy(dtype="float32")

        target_tensor = torch.tensor(target, dtype=torch.float32)
        if target_tensor.ndim == 2 and target_tensor.shape[1] == 1:
            target_tensor = target_tensor.squeeze(1)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = target_tensor

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single feature/label pair."""
        return self.features[index], self.labels[index]


class DiabetesHealthDataset(LightningDataModule):
    """End-to-end data manager for the Diabetes Health Indicators dataset.

    The module downloads the Kaggle dataset when needed, performs one-hot encoding,
    normalizes numerical features, persists processed CSV splits, and exposes
    Lightning-ready train/val/test dataloaders. Targets and feature subsets can
    be configured with semantic attribute names (e.g. ``"gender"`` expands to
    every encoded column), enabling rapid experimentation without touching the
    preprocessing pipeline.

    Example:
        >>> from pathlib import Path
        >>> dm = DiabetesHealthDataset(
        ...     data_dir=Path("data"),
        ...     target_attributes=["diagnosed_diabetes", "diabetes_risk_score"],
        ...     feature_attributes=["age", "bmi", "gender"],
        ... )
        >>> dm.prepare_data()  # downloads + preprocesses once
        >>> dm.setup("fit")    # materializes train/val datasets
        >>> batch = next(iter(dm.train_dataloader()))
    """

    KAGGLE_DS_NAME: str = "mohankrishnathalla/diabetes-health-indicators-dataset"
    POSSIBLE_TARGET_ATTRIBUTES: list[str] = ["diabetes_risk_score", "diagnosed_diabetes", "diabetes_stage"]
    CATEGORICAL_ATTRIBUTES: list[str] = [
        "gender",
        "ethnicity",
        "education_level",
        "income_level",
        "employment_status",
        "smoking_status",
        "diabetes_stage",
    ]
    CATEGORICAL_TARGET_ATTRIBUTES: list[str] = ["diagnosed_diabetes", "diabetes_stage"]
    PROCESSED_COLUMNS: list[str] = [
        "age",
        "alcohol_consumption_per_week",
        "physical_activity_minutes_per_week",
        "diet_score",
        "sleep_hours_per_day",
        "screen_time_hours_per_day",
        "family_history_diabetes",
        "hypertension_history",
        "cardiovascular_history",
        "bmi",
        "waist_to_hip_ratio",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "cholesterol_total",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
        "glucose_fasting",
        "glucose_postprandial",
        "insulin_level",
        "hba1c",
        "diabetes_risk_score",
        "diagnosed_diabetes",
        "gender_male",
        "gender_other",
        "ethnicity_black",
        "ethnicity_hispanic",
        "ethnicity_other",
        "ethnicity_white",
        "education_level_highschool",
        "education_level_no_formal",
        "education_level_postgraduate",
        "income_level_low",
        "income_level_lower-middle",
        "income_level_middle",
        "income_level_upper-middle",
        "employment_status_retired",
        "employment_status_student",
        "employment_status_unemployed",
        "smoking_status_former",
        "smoking_status_never",
        "diabetes_stage_no_diabetes",
        "diabetes_stage_pre-diabetes",
        "diabetes_stage_type_1",
        "diabetes_stage_type_2",
    ]

    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.1,
        target_attributes: Sequence[str] | str = "diagnosed_diabetes",
        feature_attributes: Sequence[str] | str | None = None,
    ) -> None:
        """Initialize data module configuration.

        Args:
            data_dir: Path to the root data directory (containing raw/processed folders).
            batch_size: Batch size shared across all dataloaders.
            num_workers: Number of workers to use in each dataloader.
            pin_memory: Whether loaders should pin memory for CUDA training.
            val_split: Fraction of the training data to reserve for validation.
            target_attributes: Supervised target attribute(s) to predict/stratify (defaults to 'diagnosed_diabetes').
            feature_attributes: Optional feature attribute(s) to retain after preprocessing. Defaults to all features.
        """
        super().__init__()

        if not 0 < val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")

        data_path = Path(data_dir)
        self.data_dir = data_path.parent if data_path.name in {"raw", "processed"} else data_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.target_attributes = self._normalize_target_attributes(target_attributes)
        self.feature_attributes = self._normalize_feature_attributes(feature_attributes)
        self.stratification_attributes = self._derive_stratification_attributes()

        self.train_dataset: Optional[DiabetesTabularDataset] = None
        self.val_dataset: Optional[DiabetesTabularDataset] = None
        self.test_dataset: Optional[DiabetesTabularDataset] = None
        self.target_columns: Optional[list[str]] = None
        self.feature_columns: Optional[list[str]] = None

    def prepare_data(self) -> None:
        """Prepare the data by downloading and preprocessing."""
        raw_dir = self.data_dir / "raw"
        processed_dir = self.data_dir / "processed"
        logger.info("Preparing dataset artifacts in {}", self.data_dir)
        self.__download_data(raw_dir)
        self.__preprocess_data(raw_dir, processed_dir)
        logger.info("Finished preparing processed data under {}", processed_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the requested Lightning stage."""
        self._ensure_processed_data()

        if stage in (None, "fit", "validate"):
            if self.train_dataset is None or self.val_dataset is None:
                logger.info(
                    "Configuring training and validation datasets (val_split={})",
                    self.val_split,
                )
                train_source_df = self._load_split("train_data.csv")
                target_columns = self._ensure_target_columns(train_source_df.columns)
                feature_columns = self._resolve_feature_columns(train_source_df.columns, target_columns)
                stratify_series = self._build_stratify_series(train_source_df, self.stratification_attributes)

                train_df, val_df = train_test_split(
                    train_source_df,
                    test_size=self.val_split,
                    random_state=constants.SEED,
                    stratify=stratify_series,
                )

                self.train_dataset = DiabetesTabularDataset(train_df, target_columns, feature_columns)
                self.val_dataset = DiabetesTabularDataset(val_df, target_columns, feature_columns)

        if stage in (None, "test", "predict") and self.test_dataset is None:
            dataset_label = "test" if stage in (None, "test") else "predict"
            logger.info("Configuring {} dataset", dataset_label)
            test_df = self._load_split("test_data.csv")
            target_columns = self._ensure_target_columns(test_df.columns)
            feature_columns = self._resolve_feature_columns(test_df.columns, target_columns)
            self.test_dataset = DiabetesTabularDataset(test_df, target_columns, feature_columns)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the train dataloader.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the validation dataloader.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting the test dataloader.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader (mirrors the test loader)."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('predict') before requesting the prediction dataloader.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def preprocess(self, output_folder: Optional[Path] = None) -> None:
        """Run preprocessing using the configured directories.

        Args:
            output_folder: Optional override for the processed data directory.
        """
        raw_dir = self.data_dir / "raw"
        processed_dir = output_folder or (self.data_dir / "processed")
        self.__preprocess_data(raw_dir, processed_dir)

    def select_features(self, feature_attributes: Sequence[str] | str | None) -> None:
        """Update the feature subset used when constructing datasets."""
        logger.info("Updating feature selection to {}", feature_attributes)
        self.feature_attributes = self._normalize_feature_attributes(feature_attributes)
        self.feature_columns = None

    def __len__(self) -> int:
        """Return the length of the training dataset if available."""
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delegate indexing to the underlying training dataset."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before indexing the dataset.")
        return self.train_dataset[index]

    def __download_data(self, dest_path: Path = Path("data/raw")) -> None:
        """Download the data if it doesn't exist."""
        if dest_path.exists() and any(dest_path.iterdir()):
            logger.info("Raw data already present at {}, skipping download.", dest_path)
            return

        logger.info("Downloading '{}' to {}", self.KAGGLE_DS_NAME, dest_path)
        cache_path = kagglehub.dataset_download(self.KAGGLE_DS_NAME, force_download=True)

        # Create destination directory if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)

        # Move all files from cache to the destination folder
        for file in os.listdir(cache_path):
            shutil.move(os.path.join(cache_path, file), os.path.join(dest_path, file))
        logger.info("Completed download and extracted files to {}", dest_path)

    def __preprocess_data(
        self, raw_data_dir: Path = Path("data/raw"), output_folder: Path = Path("data/processed")
    ) -> None:
        """Preprocess the raw data and save it to the output folder."""
        logger.info("Preprocessing raw data from {} into {}", raw_data_dir, output_folder)
        table = csv.read_csv(raw_data_dir / "diabetes_dataset.csv")
        df = table.to_pandas(self_destruct=True)

        # Find categorical columns
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Fix column names by removing spaces and special characters
        df_encoded.columns = df_encoded.columns.str.replace(" ", "_").str.lower()

        # Split data into train and test sets
        stratify_series = self._build_stratify_series(df_encoded, self.stratification_attributes)
        train_df, test_df = train_test_split(
            df_encoded, test_size=constants.TEST_SIZE, random_state=constants.SEED, stratify=stratify_series
        )
        logger.info(
            "Split dataframe into {} training rows and {} test rows",
            len(train_df),
            len(test_df),
        )

        # Compute training standardization parameters and normalize splits
        numerical_feature_columns = [
            col
            for col in train_df.columns
            if not any(class_col in col for class_col in self.POSSIBLE_TARGET_ATTRIBUTES)
            and not any(cat_col in col for cat_col in self.CATEGORICAL_ATTRIBUTES)
        ]

        if numerical_feature_columns:
            logger.info("Normalizing {} numerical feature columns", len(numerical_feature_columns))
            # Cast once to float32 so normalization uses consistent dtype.
            dtype_map = {col: "float32" for col in numerical_feature_columns}
            train_df = train_df.astype(dtype_map)
            test_df = test_df.astype(dtype_map)

            numeric_means = train_df[numerical_feature_columns].mean()
            numeric_stds = train_df[numerical_feature_columns].std().replace(0, 1)

            train_normalized = (train_df[numerical_feature_columns] - numeric_means) / numeric_stds
            test_normalized = (test_df[numerical_feature_columns] - numeric_means) / numeric_stds

            train_df.loc[:, numerical_feature_columns] = train_normalized
            test_df.loc[:, numerical_feature_columns] = test_normalized

            std_params_df = pd.DataFrame(
                {
                    "feature": numeric_means.index,
                    "mean": numeric_means.values,
                    "std": numeric_stds.values,
                }
            )
        else:
            logger.info("No numerical feature columns detected; skipping normalization step")
            std_params_df = pd.DataFrame(columns=["feature", "mean", "std"])

        # Save processed data
        os.makedirs(output_folder, exist_ok=True)
        train_df.to_csv(output_folder / "train_data.csv", index=False)
        test_df.to_csv(output_folder / "test_data.csv", index=False)
        std_params_df.to_csv(output_folder / "standardization_params.csv", index=False)
        logger.info("Persisted processed splits to {}", output_folder)

    def _normalize_target_attributes(self, target_attributes: Sequence[str] | str) -> list[str]:
        """Coerce user input into a non-empty list of attribute names."""
        if isinstance(target_attributes, str):
            attributes = [target_attributes]
        else:
            attributes = list(target_attributes)

        attributes = [attr for attr in attributes if attr]
        if not attributes:
            raise ValueError("target_attributes must contain at least one attribute name.")

        return attributes

    def _normalize_feature_attributes(self, feature_attributes: Sequence[str] | str | None) -> Optional[list[str]]:
        """Normalize optional feature attribute input."""
        if feature_attributes is None:
            return None

        if isinstance(feature_attributes, str):
            attributes = [feature_attributes]
        else:
            attributes = list(feature_attributes)

        attributes = [attr for attr in attributes if attr]
        if not attributes:
            raise ValueError("feature_attributes must contain at least one attribute name when provided.")

        return attributes

    def _derive_stratification_attributes(self) -> list[str]:
        """Select categorical targets that support stratification."""
        categorical = set(self.CATEGORICAL_TARGET_ATTRIBUTES)
        return [attr for attr in self.target_attributes if attr in categorical]

    def _resolve_feature_columns(self, available_columns: Sequence[str], target_columns: Sequence[str]) -> list[str]:
        """Resolve the concrete feature columns retained for model inputs."""
        if self.feature_columns is not None:
            return self.feature_columns

        if self.feature_attributes is None:
            features = [column for column in available_columns if column not in target_columns]
            if not features:
                raise ValueError("No feature columns remain after removing targets.")
            self.feature_columns = features
            return self.feature_columns

        resolved: list[str] = []
        for attribute in self.feature_attributes:
            resolved.extend(self._resolve_columns(available_columns, attribute))

        features = [column for column in resolved if column not in target_columns]
        if not features:
            raise ValueError("Feature attribute selection removed all available feature columns.")

        seen: set[str] = set()
        self.feature_columns = [column for column in features if not (column in seen or seen.add(column))]
        return self.feature_columns

    def _ensure_target_columns(self, available_columns: Sequence[str]) -> list[str]:
        """Resolve and cache the concrete target columns present in the data."""
        if self.target_columns is None:
            resolved: list[str] = []
            for attribute in self.target_attributes:
                resolved.extend(self._resolve_columns(available_columns, attribute))
            # Preserve order but drop duplicates
            seen: set[str] = set()
            self.target_columns = [col for col in resolved if not (col in seen or seen.add(col))]
        return self.target_columns

    def _build_stratify_series(
        self, df: pd.DataFrame, attributes: Optional[Sequence[str]] = None
    ) -> Optional[pd.Series]:
        """Return a 1D series for stratified splitting or None if not applicable."""
        attributes = [attr for attr in (attributes or []) if attr]
        if not attributes:
            return None

        attribute_labels: list[pd.Series] = []
        for attribute in attributes:
            columns = self._resolve_columns(df.columns, attribute)
            if len(columns) == 1:
                attribute_labels.append(df[columns[0]])
                continue

            encoded = df[columns].copy()
            base_column = f"{self._normalize_attribute_name(attribute)}__base_placeholder"
            encoded[base_column] = (encoded.sum(axis=1) == 0).astype(int)
            attribute_labels.append(encoded.idxmax(axis=1))

        if len(attribute_labels) == 1:
            return attribute_labels[0]

        combined = pd.Series(
            data=[tuple(values) for values in zip(*(series.astype(str) for series in attribute_labels))],
            index=df.index,
        )
        return combined

    def _resolve_columns(self, available_columns: Sequence[str], attribute: str) -> list[str]:
        """Map an attribute name to the actual encoded columns in the dataframe."""
        normalized_attribute = self._normalize_attribute_name(attribute)
        column_lookup = {self._normalize_attribute_name(column_name): column_name for column_name in available_columns}

        if normalized_attribute in column_lookup:
            return [column_lookup[normalized_attribute]]

        prefix = f"{normalized_attribute}_"
        matched = sorted(
            column_name
            for column_name in available_columns
            if self._normalize_attribute_name(column_name).startswith(prefix)
        )
        if matched:
            return matched

        raise ValueError(f"Attribute '{attribute}' could not be resolved to any data columns.")

    @staticmethod
    def _normalize_attribute_name(attribute: str) -> str:
        """Normalize attribute names using the same rules applied to column headers."""
        return attribute.strip().lower().replace(" ", "_")

    def _ensure_processed_data(self) -> None:
        """Create processed data if it is missing."""
        processed_data_dir = self.data_dir / "processed"
        if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
            self.prepare_data()

    def _load_split(self, filename: str) -> pd.DataFrame:
        """Load one of the processed CSV splits into a DataFrame."""
        processed_data_dir = self.data_dir / "processed"
        return csv.read_csv(processed_data_dir / filename).to_pandas(self_destruct=True)


def prepare_dataset(data_path: Path) -> None:
    """CLI helper to prepare raw data into processed splits.

    Args:
        data_path: Base directory that contains the raw subfolder.
    """
    logger.info("Preparing data via CLI for base path {}", data_path)
    dataset = DiabetesHealthDataset(data_path)
    dataset.prepare_data()


if __name__ == "__main__":
    typer.run(prepare_dataset)
