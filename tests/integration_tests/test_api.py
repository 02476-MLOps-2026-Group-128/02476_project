import pytest
from fastapi.testclient import TestClient

from diabetic_classification.api import app

PREDICT_DIAGNOSED_DIABETES_MLP_FEATURE_SET1_URL = "/predict/diagnosed_diabetes/MLP/feature_set1/"


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that properly handles the lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


# Sample feature set used across tests
SAMPLE_FEATURES = {
    "age": 26,
    "alcohol_consumption_per_week": 3,
    "physical_activity_minutes_per_week": 150,
    "sleep_hours_per_day": 7,
    "screen_time_hours_per_day": 8,
    "family_history_diabetes": 1,
    "bmi": 23.9,
    "systolic_bp": 122,
    "diastolic_bp": 68,
    "heart_rate": 64,
    "gender_female": 0,
    "gender_male": 1,
    "gender_other": 0,
    "ethnicity_asian": 0,
    "ethnicity_black": 0,
    "ethnicity_hispanic": 0,
    "ethnicity_other": 0,
    "ethnicity_white": 1,
    "education_level_graduate": 1,
    "education_level_highschool": 0,
    "education_level_no_formal": 0,
    "education_level_postgraduate": 0,
    "income_level_high": 0,
    "income_level_low": 0,
    "income_level_lower-middle": 0,
    "income_level_middle": 1,
    "income_level_upper-middle": 0,
    "employment_status_employed": 1,
    "employment_status_retired": 0,
    "employment_status_student": 0,
    "employment_status_unemployed": 0,
    "smoking_status_current": 0,
    "smoking_status_former": 0,
    "smoking_status_never": 1,
    "insulin_level": 3.0,
}


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_list_models(client):
    response = client.get("/models/")
    assert response.status_code == 200
    data = response.json()
    assert data is not None and len(data["models"]) > 0

    # Check that the model doesn't include the actual model object
    for problem, model_types in data["models"].items():
        for model_type, feature_sets in model_types.items():
            for feature_set, model_info in feature_sets.items():
                assert "model" not in model_info


def test_predict(client):
    features = SAMPLE_FEATURES.copy()

    response = client.post(PREDICT_DIAGNOSED_DIABETES_MLP_FEATURE_SET1_URL, json=features)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_missing_feature(client):
    features = SAMPLE_FEATURES.copy()
    del features["physical_activity_minutes_per_week"]

    response = client.post(PREDICT_DIAGNOSED_DIABETES_MLP_FEATURE_SET1_URL, json=features)
    assert response.status_code == 400
    data = response.json()
    assert "physical_activity_minutes_per_week" in data["detail"]


def test_predict_invalid_model(client):
    features = SAMPLE_FEATURES.copy()

    response = client.post("/predict/invalid_problem/MLP/feature_set1/", json=features)
    assert response.status_code == 422  # Validation error for invalid enum


def test_predict_unexpected_feature(client):
    """Test that unexpected features are rejected."""
    features = SAMPLE_FEATURES.copy()
    features["unknown_feature"] = 123.45

    response = client.post(PREDICT_DIAGNOSED_DIABETES_MLP_FEATURE_SET1_URL, json=features)
    assert response.status_code == 400
    data = response.json()
    assert "unknown_feature" in data["detail"]


def test_predict_probabilities_valid(client):
    """Test that prediction probabilities are valid."""
    features = SAMPLE_FEATURES.copy()

    response = client.post(PREDICT_DIAGNOSED_DIABETES_MLP_FEATURE_SET1_URL, json=features)
    assert response.status_code == 200
    data = response.json()

    probs = data["probabilities"]
    assert "No diabetes" in probs
    assert "Diabetes" in probs

    # Probabilities should be between 0 and 1
    assert 0 <= probs["No diabetes"] <= 1
    assert 0 <= probs["Diabetes"] <= 1

    # Probabilities should sum to approximately 1
    assert abs(probs["No diabetes"] + probs["Diabetes"] - 1.0) < 1e-6
