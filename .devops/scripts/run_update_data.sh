#!/bin/bash
set -e

uv run .devops/python_scripts/enrich_db.py

uv run .devops/python_scripts/clear_user_inputs.py

cd data

git add enriched.dvc

git commit -m "Update enriched dataset after data update."

uv run dvc push