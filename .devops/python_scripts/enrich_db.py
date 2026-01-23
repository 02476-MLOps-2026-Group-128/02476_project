import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import typer
from google.cloud import storage
from loguru import logger


def enrich_data_with_user_input() -> Path:
    """
    Enrich the dataset with prediction results.

    Performs the following steps:

    1. Fetches from the bucket the current enriched dataset as well as new user-provided
    rows

    2. Transforms them back to the original dataset schema

    3. Appends them to the dataset

    4. Overwrites the local copy of the enriched dataset.
    """
    tmp_dir = tempfile.mkdtemp()

    storage_client = storage.Client()
    bucket_name = os.environ.get("DATA_STORAGE_BUCKET_NAME")
    if bucket_name:
        bucket = storage_client.bucket(bucket_name)
        prefix = "enriched/new_rows/"
        blobs = bucket.list_blobs(prefix=prefix)
        new_rows = []

        for new_row_blob in blobs:
            relative_path = os.path.relpath(new_row_blob.name, prefix)
            local_file_path = Path(tmp_dir) / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            new_row_blob.download_to_filename(local_file_path)
            with open(local_file_path, "r") as json_file:
                row_data = json.load(json_file)
                new_rows.append(map_input_to_dataset(row_data))
        try:
            dataset_blob = bucket.blob("enriched/diabetes_dataset.csv")
            dataset_blob.download_to_filename(Path(tmp_dir) / "diabetes_dataset.csv")
        except Exception as e:
            logger.warning(f"Failed to download the dataset for enrichment: {e}")
            return

        dataset_path = Path(tmp_dir) / "diabetes_dataset.csv"
        enriched_dataset = pd.read_csv(dataset_path)
        for new_row in new_rows:
            enriched_dataset = pd.concat([enriched_dataset, pd.DataFrame([new_row])], ignore_index=True)
        enriched_dataset.to_csv(dataset_path, index=False)

        shutil.copy2(dataset_path, Path.cwd() / "data" / "enriched" / "diabetes_dataset.csv")
        return dataset_path

    else:
        logger.warning("Env var DATA_STORAGE_BUCKET_NAME not set. Skipping enrichment of dataset.")
        return


def map_input_to_dataset(user_input: dict[str, float]) -> dict:
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
        "age": user_input.get("age"),
        "alcohol_consumption_per_week": user_input.get("alcohol_consumption_per_week"),
        "physical_activity_minutes_per_week": user_input.get("physical_activity_minutes_per_week"),
        "sleep_hours_per_day": user_input.get("sleep_hours_per_day"),
        "screen_time_hours_per_day": user_input.get("screen_time_hours_per_day"),
        "family_history_diabetes": user_input.get("family_history_diabetes"),
        "bmi": user_input.get("bmi"),
        "systolic_bp": user_input.get("systolic_bp"),
        "diastolic_bp": user_input.get("diastolic_bp"),
        "heart_rate": user_input.get("heart_rate"),
        "insulin_level": user_input.get("insulin_level"),
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
        "diabetes_risk_score": user_input.get("probabilities"),
        "diabetes_stage": None,
        "diagnosed_diabetes": user_input.get("prediction"),
    }

    # Process One-Hot groups to retrieve the original categorical string
    for original_key, sub_features in one_hot_groups.items():
        # Find which column has the value 1 (or 1.0/True)
        active_feature = next((f for f in sub_features if user_input.get(f) == 1), None)

        if active_feature:
            # Strip the prefix (e.g., "gender_" -> "male")
            result[original_key] = active_feature.replace(f"{original_key}_", "")
        else:
            result[original_key] = None

    return result


if __name__ == "__main__":
    typer.run(enrich_data_with_user_input)
