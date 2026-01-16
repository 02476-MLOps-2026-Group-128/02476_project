from __future__ import annotations

from contextlib import asynccontextmanager
from http import HTTPStatus
import json
import os

import torch
from fastapi import FastAPI
from enum import Enum

from diabetic_classification.model import TabularMLP


# Global variables initialized in lifespan
device: torch.device
fs1_mlp: TabularMLP
model_registry: dict[str, dict[str, dict[str, dict]]]


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
    global model_registry, fs1_mlp, device
    print("Loading feature sets")
    # Load feature sets from configs/feature_sets
    available_feature_sets = os.listdir("configs/feature_sets")
    print(f"Available feature sets: {available_feature_sets}")
    # Load feature sets from configs/feature_sets
    feature_sets = {}
    for fs_file in available_feature_sets:
        if fs_file.endswith(".json"):
            fs_name = fs_file[:-5]  # Remove .json extension
            with open(os.path.join("configs/feature_sets", fs_file), "r") as f:
                feature_sets[fs_name] = json.load(f)

    print("Loading models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Binary models (diagnosed diabetes)
    fs1_mlp = TabularMLP(
        input_dim=35,
        hidden_dims=(128, 64),
        dropout=0.2,  # Must match training config
        output_dim=1,
    )
    fs1_mlp.to(device)
    fs1_mlp.load_state_dict(
        torch.load(
            "models/diagnosed_diabetes/MLP/feature_set1/best_model.pt",
            map_location=device,
        )
    )
    fs1_mlp.eval()

    model_registry = {
        "diagnosed_diabetes": {
            "MLP": {
                "feature_set1": {
                    "model": fs1_mlp,
                    "input_dim": 35,
                    "output_dim": 1,
                    "task_type": TaskType.BINARY_CLASSIFICATION,
                    "features": feature_sets["feature_set1"]
                },
            }
        }
    }
    yield

    print("Cleaning up")
    del fs1_mlp, device, model_registry


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {
        "message": "Diabetic Classification API is running.",
        "status-code": HTTPStatus.OK,
    }


@app.get("/models/")
def list_models():
    """List available models in the registry."""
    # Get keys of model_registry but without the actual models
    registry_overview = {
        problem: {
            model_type: {
                feature_set: {
                    key: value for key, value in model_info.items() if key != "model"
                }
                for feature_set, model_info in feature_sets.items()
            }
            for model_type, feature_sets in model_types.items()
        }
        for problem, model_types in model_registry.items()
    }
    return {
        "models": registry_overview,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


@app.post("/predict/{problem_type}/{model_type}/{feature_set}/")
def predict(
        problem_type: ProblemType,
        model_type: ModelType,
        feature_set: FeatureSet,
        features: dict[str, float],
):
    """Make a prediction using the specified model."""
    if problem_type.value not in model_registry:
        return {
            "error": f"Problem type '{problem_type}' not found.",
            "status-code": HTTPStatus.NOT_FOUND,
        }
    if model_type.value not in model_registry[problem_type.value]:
        return {
            "error": f"Model type '{model_type}' not found for problem '{problem_type}'.",
            "status-code": HTTPStatus.NOT_FOUND,
        }
    if feature_set.value not in model_registry[problem_type.value][model_type.value]:
        return {
            "error": f"Feature set '{feature_set}' not found for model type '{model_type}' and problem '{problem_type}'.",
            "status-code": HTTPStatus.NOT_FOUND,
        }

    model_info = model_registry[problem_type.value][model_type.value][feature_set.value]
    model = model_info["model"]
    input_dim = model_info["input_dim"]
    output_dim = model_info["output_dim"]
    task_type = model_info["task_type"]
    expected_features = model_info["features"]

    # Check if all expected features are present
    missing_features = set(expected_features) - set(features.keys())
    if missing_features:
        return {
            "error": f"Missing required features: {sorted(missing_features)}",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    # Check for unexpected features
    unexpected_features = set(features.keys()) - set(expected_features)
    if unexpected_features:
        return {
            "error": f"Unexpected features provided: {sorted(unexpected_features)}",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    # Reorder features according to expected order
    ordered_features = [features[feat] for feat in expected_features]

    if len(ordered_features) != input_dim:
        return {
            "error": f"Expected {input_dim} features, but got {len(ordered_features)}.",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    with torch.no_grad():
        input_tensor = torch.tensor(
            [ordered_features], dtype=torch.float32).to(device)
        logits = model(input_tensor)
        if task_type == TaskType.BINARY_CLASSIFICATION:
            prob_class_1 = float(torch.sigmoid(logits).cpu().item())
            prob_class_0 = 1 - prob_class_1
            probs = {"No diabetes": prob_class_0, "Diabetes": prob_class_1}
            prediction = 1 if prob_class_1 >= 0.5 else 0
        else:
            return {
                "error": f"Unsupported task type '{task_type}'.",
                "status-code": HTTPStatus.BAD_REQUEST,
            }

    return {
        "prediction": prediction,
        "probabilities": probs,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
