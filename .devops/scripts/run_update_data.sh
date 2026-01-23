#!/bin/bash
set -e

uv run .devops/python_scripts/enrich_db.py

uv run .devops/python_scripts/clear_user_inputs.py

cd data

uv run dvc add enriched/diabetes_dataset.csv

git add enriched.dvc

git commit -m "Update enriched dataset after data update."

uv run dvc push