import json
import random

from locust import HttpUser, between, task

SAMPLE_PREDICTION_PAYLOAD = {
    "age": 45.0,
    "alcohol_consumption_per_week": 2.0,
    "physical_activity_minutes_per_week": 150.0,
    "sleep_hours_per_day": 7.0,
    "screen_time_hours_per_day": 3.0,
    "family_history_diabetes": 1.0,
    "bmi": 27.5,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "heart_rate": 72.0,
    "gender_female": 0.0,
    "gender_male": 1.0,
    "gender_other": 0.0,
    "ethnicity_asian": 0.0,
    "ethnicity_black": 0.0,
    "ethnicity_hispanic": 0.0,
    "ethnicity_other": 0.0,
    "ethnicity_white": 1.0,
    "education_level_graduate": 0.0,
    "education_level_highschool": 1.0,
    "education_level_no_formal": 0.0,
    "education_level_postgraduate": 0.0,
    "income_level_high": 0.0,
    "income_level_low": 0.0,
    "income_level_lower-middle": 0.0,
    "income_level_middle": 1.0,
    "income_level_upper-middle": 0.0,
    "employment_status_employed": 1.0,
    "employment_status_retired": 0.0,
    "employment_status_student": 0.0,
    "employment_status_unemployed": 0.0,
    "smoking_status_current": 0.0,
    "smoking_status_former": 0.0,
    "smoking_status_never": 1.0,
    "insulin_level": 15.2,
}


class TestUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def fetch_model_registry(self):
        self.client.get("/models/")

    @task
    def predict(self):
        self.client.post(
            "/predict/diagnosed_diabetes/MLP/feature_set1/",
            data=json.dumps(SAMPLE_PREDICTION_PAYLOAD),
            headers={"Content-Type": "application/json"},
        )

    @task
    def health_check(self):
        self.client.get("/")
