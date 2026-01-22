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

import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException
from google.cloud import storage
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic_settings import BaseSettings, SettingsConfigDict

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
feature_sets: dict[str, list[str]]


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


# Prometheus metrics
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds",
    "Latency of API predictions in seconds",
    ["model_type", "feature_set", "problem_type"],
)
PREDICTION_REQUEST_COUNT = Counter(
    "api_prediction_request_count",
    "Total number of prediction requests",
    ["model_type", "feature_set", "problem_type"],
)
MODEL_LOADING_TIME = Histogram(
    "api_model_loading_time_seconds",
    "Time taken to load all models during startup in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model_registry, device, feature_sets

    # 1. Determine the Base Path for Artifacts
    artifacts_gcs_uri = os.environ.get("ARTIFACTS_GCS_URI")
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI")
    gcs_uri = artifacts_gcs_uri or aip_storage_uri
    tmp_dir = None

    tmp_dir = tempfile.mkdtemp()
    base_path = Path(tmp_dir)
    if gcs_uri:
        logger.info(f"Running in cloud. Downloading artifacts from: {gcs_uri}")

        # Download everything from the GCS bucket path to our temp folder
        # Expected URI format: gs://bucket-name/path/to/artifacts/
        models_bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
        prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

        model_storage_client = storage.Client()
        model_bucket = model_storage_client.bucket(models_bucket_name)
        blobs = model_bucket.list_blobs(prefix=prefix)

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
        feature_sets_dir = base_path / "configs" / "feature_sets"
    else:
        logger.info("Running locally. Using local configuration paths.")
        base_path = Path(".")  # Current directory
        models_dir = base_path / settings.api_models_dir
        feature_sets_dir = base_path / settings.feature_sets_dir

    # 2. Load Feature Sets (Using dynamic paths)
    feature_sets = {}
    logger.info(f"Looking for feature sets in: {feature_sets_dir}")
    if feature_sets_dir.exists():
        files = os.listdir(feature_sets_dir)
        logger.info(f"Feature sets directory contents: {files}")
        for fs_file in files:
            if not fs_file.endswith(".json"):
                continue
            fs_name = fs_file[:-5]
            with open(feature_sets_dir / fs_file, "r") as f:
                feature_sets[fs_name] = json.load(f)
    else:
        logger.warning(f"Feature sets directory does not exist: {feature_sets_dir}")

    if not feature_sets:
        logger.error(f"No feature sets found in {feature_sets_dir}. Check your GCS artifact path and contents.")
        raise RuntimeError(f"No feature sets found in {feature_sets_dir}. Check your GCS artifact path and contents.")

    # 3. Discover and Load Models from Directory Structure
    logger.info("Discovering models from directory structure")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    start_model_load = time.perf_counter()
    model_registry = {}

    # Walk through the models directory: models/problem_type/model_type/feature_set/version/
    if models_dir.exists():
        for problem_path in models_dir.iterdir():
            if not problem_path.is_dir():
                continue

            problem_type = problem_path.name
            model_registry[problem_type] = {}

            for model_type_path in problem_path.iterdir():
                if not model_type_path.is_dir():
                    continue

                model_type = model_type_path.name
                model_registry[problem_type][model_type] = {}

                for feature_set_path in model_type_path.iterdir():
                    if not feature_set_path.is_dir():
                        continue

                    feature_set_name = feature_set_path.name
                    model_registry[problem_type][model_type][feature_set_name] = {}

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
                                    "feature_set": feature_set_name,
                                    "prediction_endpoint": f"/predict/{problem_type}/{model_type}/{feature_set_name}/",
                                }
                            else:
                                logger.error(f"Unknown model type: {model_type}")
                                raise ValueError(f"Model type '{model_type}' is not supported")

                            model_registry[problem_type][model_type][feature_set_name][version] = model_info

                        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                            logger.error(f"Failed to load model from {version_path}: {e}")
                            raise

    model_load_time = time.perf_counter() - start_model_load
    MODEL_LOADING_TIME.observe(model_load_time)
    logger.info(f"Model registry initialized (loading time: {model_load_time:.2f}s)")

    data_storage_uri = os.environ.get("DATA_STORAGE_URI")
    if data_storage_uri:
        logger.info(f"Data storage URI found: {data_storage_uri}")

        data_bucket_name = data_storage_uri.replace("gs://", "").split("/")[0]
        data_prefix = "/".join(data_storage_uri.replace("gs://", "").split("/")[1:])

        data_storage_client = storage.Client()
        data_bucket = data_storage_client.bucket(data_bucket_name)
        blobs = data_bucket.list_blobs(prefix=data_prefix)

        for blob in blobs:
            # Create local subdirectories if they exist in GCS
            relative_path = os.path.relpath(blob.name, data_prefix)
            local_file_path = base_path / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            if not blob.name.endswith("/"):  # Skip directory markers
                blob.download_to_filename(str(local_file_path))
                logger.debug(f"Downloaded: {blob.name} -> {local_file_path}")

        # Update paths to point to the temporary local directory
        db_dir = base_path / "enriched" / "api_models"
    else:
        logger.warning("Data storage URI not found. Data will not be enriched from new user entries.")
    yield

    logger.info("Cleaning up")
    del model_registry, device

    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def get_model_registry() -> ModelRegistry:
    """Dependency to get the model registry."""
    return model_registry


def get_device() -> torch.device:
    """Dependency to get the compute device."""
    return device


def get_feature_sets() -> dict[str, list[str]]:
    """Dependency to get the feature sets."""
    return feature_sets


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
            "list_feature_sets": "/feature-sets/",
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


@app.get("/feature-sets/")
async def list_feature_sets(
    feature_sets: dict[str, list[str]] = Depends(get_feature_sets),
) -> dict[str, Any]:
    """List available feature sets for each problem and model type."""
    return feature_sets


@app.post("/predict/{problem_type}/{model_type}/{feature_set}/")
async def predict(
    problem_type: ProblemType,
    model_type: ModelType,
    feature_set: FeatureSet,
    features: dict[str, float],
    registry: ModelRegistry = Depends(get_model_registry),
    device: torch.device = Depends(get_device),
    feature_sets: dict[str, list[str]] = Depends(get_feature_sets),
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
    expected_features = feature_sets[feature_set.value]

    # Temporary workaround to load expected features from file
    with open("configs/feature_sets/feature_set1.json", "r") as f:
        expected_features = json.load(f)
    logger.debug(f"Using model version: {version} for {problem_type.value}/{model_type.value}/{feature_set.value}")
    logger.debug(f"Feature set version: {feature_set_versions[version]}")
    logger.debug(f"Model info{model_info}")

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
    enrich_dataset(features, prediction, probs)

    PREDICTION_LATENCY.labels(
        model_type=model_type.value,
        feature_set=feature_set.value,
        problem_type=problem_type.value,
    ).observe(inference_time)
    PREDICTION_REQUEST_COUNT.labels(
        model_type=model_type.value,
        feature_set=feature_set.value,
        problem_type=problem_type.value,
    ).inc()

    return {
        "prediction": prediction,
        "probabilities": probs,
    }


def enrich_dataset(features: dict[str, float], prediction: int, probabilities: dict[str, float]) -> None:
    """Enrich the dataset with prediction results (stub function)."""
    logger.debug(
        f"Enriching dataset with features: {features}, prediction: {prediction}, probabilities: {probabilities}"
    )

    new_row = map_model_to_dataset(
        model_input=features,
        prediction=prediction,
        probabilities=probabilities,
    )

    enriched_dataset = pd.read_csv(
        "/home/mathias/Documents/academic/dtu/ml_ops/02476_project/data/enriched/diabetes_dataset.csv"
    )
    enriched_dataset = pd.concat([enriched_dataset, pd.DataFrame([new_row])], ignore_index=True)
    enriched_dataset.to_csv(
        "/home/mathias/Documents/academic/dtu/ml_ops/02476_project/data/enriched/diabetes_dataset.csv", index=False
    )
    logger.info(f"New dataset size: {len(enriched_dataset)}.")
    logger.info("Dataset enriched and saved to 'data/enriched/diabetes_dataset.csv'")


def map_model_to_dataset(model_input: dict[str, float], prediction: int, probabilities: dict[str, float]) -> dict:
    """
    Transform a model inference dictionary (One-Hot Encoded).

    Back to the original dataset schema.
    """
    # Define the groups of one-hot encoded columns
    one_hot_groups = {
        "gender": ["gender_female", "gender_male", "gender_other"],
        "ethnicity": ["ethnicity_asian", "ethnicity_black", "ethnicity_hispanic", "ethnicity_other", "ethnicity_white"],
        "education_level": [
            "education_level_graduate",
            "education_level_highschool",
            "education_level_no_formal",
            "education_level_postgraduate",
        ],
        "income_level": [
            "income_level_high",
            "income_level_low",
            "income_level_lower-middle",
            "income_level_middle",
            "income_level_upper-middle",
        ],
        "employment_status": [
            "employment_status_employed",
            "employment_status_retired",
            "employment_status_student",
            "employment_status_unemployed",
        ],
        "smoking_status": ["smoking_status_current", "smoking_status_former", "smoking_status_never"],
    }

    # Start with direct numerical/boolean mappings
    result = {
        "age": model_input.get("age"),
        "alcohol_consumption_per_week": model_input.get("alcohol_consumption_per_week"),
        "physical_activity_minutes_per_week": model_input.get("physical_activity_minutes_per_week"),
        "sleep_hours_per_day": model_input.get("sleep_hours_per_day"),
        "screen_time_hours_per_day": model_input.get("screen_time_hours_per_day"),
        "family_history_diabetes": model_input.get("family_history_diabetes"),
        "bmi": model_input.get("bmi"),
        "systolic_bp": model_input.get("systolic_bp"),
        "diastolic_bp": model_input.get("diastolic_bp"),
        "heart_rate": model_input.get("heart_rate"),
        "insulin_level": model_input.get("insulin_level"),
        # Fields not present in model JSON default to None
        "diet_score": None,
        "hypertension_history": None,
        "cardiovascular_history": None,
        "waist_to_hip_ratio": None,
        "cholesterol_total": None,
        "hdl_cholesterol": None,
        "ldl_cholesterol": None,
        "triglycerides": None,
        "glucose_fasting": None,
        "glucose_postprandial": None,
        "hba1c": None,
        "diabetes_risk_score": probabilities.get("Diabetes"),
        "diabetes_stage": None,
        "diagnosed_diabetes": prediction,
    }

    # Process One-Hot groups to retrieve the original categorical string
    for original_key, sub_features in one_hot_groups.items():
        # Find which column has the value 1 (or 1.0/True)
        active_feature = next((f for f in sub_features if model_input.get(f) == 1), None)

        if active_feature:
            # Strip the prefix (e.g., "gender_" -> "male")
            result[original_key] = active_feature.replace(f"{original_key}_", "")
        else:
            result[original_key] = None

    return result
