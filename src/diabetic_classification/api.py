from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import anyio
import pandas as pd
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.legacy.report import Report

from diabetic_classification.model import TabularMLP


class Settings(BaseSettings):
    """API configuration settings."""

    # Paths
    feature_sets_dir: str = "configs/feature_sets"
    api_models_dir: str = "models/api_models"

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()

# Type aliases for better readability
# ModelConfig: flexible dict containing model-specific parameters (varies by model type)
# ModelInfo: runtime information for loaded models
ModelRegistry = dict[str, dict[str, dict[str, dict[str, Any]]]]

# Global variables initialized in lifespan
device: torch.device
model_registry: ModelRegistry


class ProblemType(str, Enum):
    """Enumeration of problem types."""

    DIAGNOSED_DIABETES = "diagnosed_diabetes"


class FeatureSet(str, Enum):
    """Enumeration of feature sets."""

    FEATURE_SET_1 = "feature_set1"


class ModelType(str, Enum):
    """Enumeration of model types."""

    MLP = "MLP"


class TaskType(str, Enum):
    """Enumeration of task types."""

    BINARY_CLASSIFICATION = "binary_classification"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model_registry, device

    # 1. Determine the Base Path for Artifacts
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI")
    tmp_dir = None

    if aip_storage_uri:
        logger.info(f"Running in Vertex AI. Downloading weights from: {aip_storage_uri}")
        # Create a temporary directory to store downloaded artifacts
        tmp_dir = tempfile.mkdtemp()
        base_path = Path(tmp_dir)

        # Download everything from the GCS bucket path to our temp folder
        # Expected URI format: gs://bucket-name/path/to/model/v1/
        bucket_name = aip_storage_uri.replace("gs://", "").split("/")[0]
        prefix = "/".join(aip_storage_uri.replace("gs://", "").split("/")[1:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            # Create local subdirectories if they exist in GCS
            relative_path = os.path.relpath(blob.name, prefix)
            local_file_path = base_path / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            if not blob.name.endswith("/"):  # Skip directory markers
                blob.download_to_filename(str(local_file_path))
                logger.debug(f"Downloaded: {blob.name} -> {local_file_path}")

        # Update paths to point to the temporary local directory
        models_dir = base_path / "models" / "api_models"
        feature_sets_dir = base_path / "feature_sets"
    else:
        logger.info("Running locally. Using local configuration paths.")
        base_path = Path(".")  # Current directory
        models_dir = base_path / settings.api_models_dir
        feature_sets_dir = base_path / settings.feature_sets_dir

    # 2. Load Feature Sets (Using dynamic paths)
    feature_sets: dict[str, list[str]] = {}
    if feature_sets_dir.exists():
        for fs_file in os.listdir(feature_sets_dir):
            if fs_file.endswith(".json"):
                fs_name = fs_file[:-5]
                with open(feature_sets_dir / fs_file, "r") as f:
                    feature_sets[fs_name] = json.load(f)

    # 3. Discover and Load Models from Directory Structure
    logger.info("Discovering models from directory structure")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_registry = {}

    # Walk through the models directory: models/problem_type/model_type/feature_set/version/
    if models_dir.exists():
        for problem_path in models_dir.iterdir():
            if not problem_path.is_dir():
                continue

            problem_type = problem_path.name

            for model_type_path in problem_path.iterdir():
                if not model_type_path.is_dir():
                    continue

                model_type = model_type_path.name

                for feature_set_path in model_type_path.iterdir():
                    if not feature_set_path.is_dir():
                        continue

                    feature_set_name = feature_set_path.name

                    for version_path in feature_set_path.iterdir():
                        if not version_path.is_dir():
                            continue

                        version = version_path.name
                        config_file = version_path / "config.json"

                        if not config_file.exists():
                            logger.debug(f"No config.json found in {version_path}, skipping")
                            continue

                        logger.info(f"Loading {problem_type}/{model_type}/{feature_set_name}/{version}")

                        try:
                            with open(config_file, "r") as f:
                                config = json.load(f)

                            # Resolve model path relative to version directory
                            model_path = version_path / config["model_path"]
                            if not model_path.exists():
                                logger.error(f"Model file not found at {model_path}")
                                raise FileNotFoundError(f"Model file not found: {model_path}")

                            # Create model based on type
                            if model_type == ModelType.MLP.value:
                                model = TabularMLP(
                                    input_dim=config["input_dim"],
                                    hidden_dims=tuple(config["hidden_dims"]),
                                    dropout=config["dropout"],
                                    output_dim=config["output_dim"],
                                )
                                model.to(device)
                                model.load_state_dict(torch.load(str(model_path), map_location=device))
                                model.eval()

                                # Store in registry
                                model_info: dict[str, Any] = {
                                    "model": model,
                                    "input_dim": config["input_dim"],
                                    "output_dim": config["output_dim"],
                                    "task_type": TaskType[config["task_type"].upper()],
                                    "features": feature_sets.get(feature_set_name, []),
                                }
                            else:
                                logger.error(f"Unknown model type: {model_type}")
                                raise ValueError(f"Model type '{model_type}' is not supported")

                            # Ensure nested structure exists before adding
                            if problem_type not in model_registry:
                                model_registry[problem_type] = {}
                            if model_type not in model_registry[problem_type]:
                                model_registry[problem_type][model_type] = {}
                            if feature_set_name not in model_registry[problem_type][model_type]:
                                model_registry[problem_type][model_type][feature_set_name] = {}

                            model_registry[problem_type][model_type][feature_set_name][version] = model_info

                        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                            logger.error(f"Failed to load model from {version_path}: {e}")
                            raise

    logger.info("Model registry initialized")
    yield

    logger.info("Cleaning up")
    del model_registry, device

    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


app = FastAPI(lifespan=lifespan)


def get_model_registry() -> ModelRegistry:
    """Dependency to get the model registry."""
    return model_registry


def get_device() -> torch.device:
    """Dependency to get the compute device."""
    return device


@app.get("/")
async def read_root(registry: ModelRegistry = Depends(get_model_registry)) -> dict[str, Any]:
    """Enhanced health check endpoint with API status information."""
    # Count available models
    model_count = sum(len(feature_sets) for model_types in registry.values() for feature_sets in model_types.values())

    return {
        "status": "healthy",
        "message": "Diabetic Classification API is running",
        "version": "1.0.0",
        "models_loaded": model_count,
        "endpoints": {
            "health": "/",
            "list_models": "/models/",
            "predict": "/predict/{problem_type}/{model_type}/{feature_set}/",
        },
    }


@app.get("/models/")
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> dict[str, Any]:
    """List available models in the registry."""
    registry_overview: dict[str, Any] = {}
    for problem, model_types in registry.items():
        registry_overview[problem] = {}
        for model_type, feature_sets in model_types.items():
            registry_overview[problem][model_type] = {}
            for feature_set, versions in feature_sets.items():
                registry_overview[problem][model_type][feature_set] = {}
                for version, model_info in versions.items():
                    registry_overview[problem][model_type][feature_set][version] = {
                        key: value for key, value in model_info.items() if key != "model"
                    }
    return {"models": registry_overview}


@app.post("/predict/{problem_type}/{model_type}/{feature_set}/")
async def predict(
    problem_type: ProblemType,
    model_type: ModelType,
    feature_set: FeatureSet,
    features: dict[str, float],
    registry: ModelRegistry = Depends(get_model_registry),
    device: torch.device = Depends(get_device),
) -> dict[str, Any]:
    """Make a prediction using the specified model."""
    if problem_type.value not in registry:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Problem type '{problem_type}' not found.",
        )
    if model_type.value not in registry[problem_type.value]:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Model type '{model_type}' not found for problem '{problem_type}'.",
        )
    if feature_set.value not in registry[problem_type.value][model_type.value]:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Feature set '{feature_set}' not found for model type '{model_type}' and problem '{problem_type}'.",
        )

    feature_set_versions = registry[problem_type.value][model_type.value][feature_set.value]
    if not feature_set_versions:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No models available for feature set '{feature_set}'.",
        )

    version = sorted(feature_set_versions.keys())[-1]
    model_info = feature_set_versions[version]
    model = model_info["model"]
    input_dim = model_info["input_dim"]
    task_type = model_info["task_type"]
    expected_features = model_info["features"]

    # Check if all expected features are present
    missing_features = set(expected_features) - set(features.keys())
    if missing_features:
        logger.warning(
            f"Prediction failed - missing features: {sorted(missing_features)} "
            f"for {problem_type.value}/{model_type.value}/{feature_set.value}"
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Missing required features: {sorted(missing_features)}",
        )

    # Check for unexpected features
    unexpected_features = set(features.keys()) - set(expected_features)
    if unexpected_features:
        logger.warning(
            f"Prediction failed - unexpected features: {sorted(unexpected_features)} "
            f"for {problem_type.value}/{model_type.value}/{feature_set.value}"
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Unexpected features provided: {sorted(unexpected_features)}",
        )

    # Reorder features according to expected order
    ordered_features = [features[feat] for feat in expected_features]

    if len(ordered_features) != input_dim:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Expected {input_dim} features, but got {len(ordered_features)}.",
        )

    start_time = time.perf_counter()

    with torch.no_grad():
        input_tensor = torch.tensor([ordered_features], dtype=torch.float32).to(device)
        logits = model(input_tensor)
        if task_type == TaskType.BINARY_CLASSIFICATION:
            prob_class_1 = float(torch.sigmoid(logits).cpu().item())
            prob_class_0 = 1 - prob_class_1
            probs = {"No diabetes": prob_class_0, "Diabetes": prob_class_1}
            prediction = 1 if prob_class_1 >= 0.5 else 0
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Unsupported task type '{task_type}'.",
            )

    inference_time = time.perf_counter() - start_time
    logger.debug(
        f"Prediction completed in {inference_time * 1000:.2f}ms for "
        f"{problem_type.value}/{model_type.value}/{feature_set.value}"
    )
    logger.info(
        f"Prediction: {prediction}, prob_diabetes={prob_class_1:.3f} "
        f"for {problem_type.value}/{model_type.value}/{feature_set.value}"
    )

    return {
        "prediction": prediction,
        "probabilities": probs,
    }

BUCKET_NAME = "diabetes-health-indicators-dataset"

@app.get("/reports", response_class=HTMLResponse)
async def reports():
    """Get request method that returns a monitoring report."""
    reference_data, input_data = get_data_drift_data()

    # Evidently expects the target column to be named "target"
    reference_data = reference_data.rename(columns={"diagnosed_diabetes": "target"})
    input_data = input_data.rename(columns={"diagnosed_diabetes": "target"})

    # Find columns that are 100% null in input_data
    empty_cols = input_data.columns[input_data.isnull().all()]

    # Drop them from both so the schemas match
    reference_data = reference_data.drop(columns=empty_cols)
    input_data = input_data.drop(columns=empty_cols)

    # Generate Evidently report and save as HTML
    data_drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset()])
    data_drift_report.run(current_data=input_data, reference_data=reference_data)
    data_drift_report.save_html("datadrift.html")

    # Read the generated HTML report and return as response
    async with await anyio.open_file("datadrift.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)

def get_data_drift_data(n = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the enriched diabetes dataset from GCP bucket and split into train and input data."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("enriched/diabetes_dataset.csv")
    blob.download_to_filename("input_data.csv")
    data = pd.read_csv("input_data.csv")

    # The first 100.000 rows were used as training data
    train_length = 100000
    train_data = data.iloc[:train_length]

    # The remaining rows were user input
    input_data = data.iloc[train_length:]

     # If n is specified, return only the last n rows of input_data
    if n is None:
        return train_data, input_data

    return train_data, input_data.tail(n)