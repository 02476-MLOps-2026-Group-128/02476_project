import pandas as pd
import json
import os
import anyio
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.legacy.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from pathlib import Path

BUCKET_NAME = "diabetes-monitoring"

def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    data_drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save("data_drift_report.html")

def lifespan(app: FastAPI):
    """Load the training data before the application starts."""
    global training_data

    # Download the training data from the GCP bucket
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob("processed_train_data.csv")
    blob.download_to_filename(blob.name)

    # Load the training data into a DataFrame
    training_data = pd.read_csv(blob.name)

    yield

    del training_data

app = FastAPI(lifespan=lifespan)

def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("prediction_*.json")

    # Sort files based on when they were created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    feature_input, prediction = list[float], list[float]
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            feature_input.append(data["input"])
            prediction.append(data["prediction"])
    dataframe = pd.DataFrame({"content": feature_input, "prediction": prediction})
    dataframe["target"] = dataframe["prediction"]
    return dataframe


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="prediction_")
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    for blob in latest_blobs:
        blob.download_to_filename(blob.name)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5) -> HTMLResponse:
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)