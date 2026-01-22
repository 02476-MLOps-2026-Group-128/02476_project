# Infrastructure & Deployment

This directory contains the Terraform setup for deploying the FastAPI inference API and managing GCP resources for this project.

## Overview

The Terraform configuration provisions:
- Cloud Run service for serving the FastAPI API
- Artifact Registry for Docker images
- Google Cloud Storage (GCS) bucket for model registry and feature sets
- IAM roles and service accounts for secure access

## Prerequisites
- GCP project with billing enabled
- Terraform installed (>=1.0)
- gcloud CLI installed and authenticated
- Artifact Registry and Cloud Run APIs enabled
- Docker installed (for local builds)

## Setup & Usage

1. **Configure variables:**
   - Edit `terraform.tfvars` or use `-var` flags to set variables like `container_image_tag`, `gcp_project`, `region`, etc.

2. **Initialize Terraform:**
   ```bash
   cd infrastructure
   terraform init
   ```

3. **Plan and apply deployment:**
   ```bash
   terraform plan -var="container_image_tag=<your-tag>"
   terraform apply -var="container_image_tag=<your-tag>"
   ```
   - Use a unique image tag for each deployment to trigger a new Cloud Run revision.

4. **Build and push Docker images:**
   - Use Cloud Build or local Docker to build and push images to Artifact Registry.
   - Example (Cloud Build):
     ```bash
     gcloud builds submit --config ../cloudbuild.yaml .. --substitutions=SHORT_SHA=$(git rev-parse --short HEAD)
     ```

5. **Access the API:**
   - After deployment, find the Cloud Run service URL in Terraform outputs or GCP Console.

## Key Files
- `main.tf`: Main Terraform configuration
- `variables.tf`: Variable definitions
- `terraform.tfvars`: Variable values
- `../cloudbuild.yaml`: CI/CD build pipeline
    - HOWEVER, as Cloud Build doesn't keep a cache between runs they take very long (20 mins). It is usually better to build the images locally and push to the registry:
    ```sh
    docker build -f dockerfiles/fastapi.dockerfile -t europe-west4-docker.pkg.dev/diabetic-classification-484510/container-registry/fastapi:latest -t europe-west4-docker.pkg.dev/diabetic-classification-484510/container-registry/fastapi:$(git rev-parse --short HEAD) .

    docker push europe-west4-docker.pkg.dev/diabetic-classification-484510/container-registry/fastapi:latest

    docker push europe-west4-docker.pkg.dev/diabetic-classification-484510/container-registry/fastapi:$(git rev-parse --short HEAD)
    ```
- `../dockerfiles/fastapi.dockerfile`: FastAPI Docker build

## Notes
- All infrastructure changes should be managed via Terraform for reproducibility.
- Existing GCS buckets can be imported into Terraform state using `terraform import`.
- Environment variables (e.g., `ARTIFACTS_GCS_URI`) are set in Cloud Run via Terraform.
- For troubleshooting, check Cloud Run logs and Terraform output.
