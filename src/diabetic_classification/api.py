from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException
from google.cloud import storage
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from diabetic_classification.model import TabularMLP


class Settings(BaseSettings):
    """API configuration settings."""

    # Paths
    feature_sets_dir: str = "configs/feature_sets"
    model_configs_path: str = "configs/models.json"

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
        model_configs_path = base_path / "models.json"
        feature_sets_dir = base_path / "feature_sets"
    else:
        logger.info("Running locally. Using local configuration paths.")
        base_path = Path(".")  # Current directory
        model_configs_path = base_path / settings.model_configs_path
        feature_sets_dir = base_path / settings.feature_sets_dir

    # 2. Load Feature Sets (Using dynamic paths)
    feature_sets: dict[str, list[str]] = {}
    if feature_sets_dir.exists():
        for fs_file in os.listdir(feature_sets_dir):
            if fs_file.endswith(".json"):
                fs_name = fs_file[:-5]
                with open(feature_sets_dir / fs_file, "r") as f:
                    feature_sets[fs_name] = json.load(f)

    # 3. Load Model Configurations
    with open(model_configs_path, "r") as f:
        model_configs = json.load(f)

    logger.info("Loading models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models based on configuration
    model_registry = {}
    for problem_type, model_types in model_configs.items():
        model_registry[problem_type] = {}
        for model_type, feature_set_configs in model_types.items():
            model_registry[problem_type][model_type] = {}
            for feature_set_name, version_configs in feature_set_configs.items():
                model_registry[problem_type][model_type][feature_set_name] = {}
                for version, config in version_configs.items():
                    logger.info(f"Loading {problem_type}/{model_type}/{feature_set_name}/{version}")

                    # Create model based on type
                    if model_type == ModelType.MLP.value:
                        model = TabularMLP(
                            input_dim=config["input_dim"],
                            hidden_dims=tuple(config["hidden_dims"]),
                            dropout=config["dropout"],
                            output_dim=config["output_dim"],
                        )
                        model.to(device)
                        model.load_state_dict(torch.load(config["model_path"], map_location=device))
                        model.eval()

                        # Store in registry
                        model_info: dict[str, Any] = {
                            "model": model,
                            "input_dim": config["input_dim"],
                            "output_dim": config["output_dim"],
                            "task_type": TaskType[config["task_type"].upper()],
                            "features": feature_sets[feature_set_name],
                        }
                    else:
                        logger.error(f"Unknown model type: {model_type}")
                        raise ValueError(f"Model type '{model_type}' is not supported")

                    model_registry[problem_type][model_type][feature_set_name][version] = model_info

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
    # Get keys of model_registry but without the actual models
    registry_overview = {
        problem: {
            model_type: {
                feature_set: {key: value for key, value in model_info.items() if key != "model"}
                for feature_set, model_info in feature_sets.items()
            }
            for model_type, feature_sets in model_types.items()
        }
        for problem, model_types in registry.items()
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

    model_info = registry[problem_type.value][model_type.value][feature_set.value]
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
