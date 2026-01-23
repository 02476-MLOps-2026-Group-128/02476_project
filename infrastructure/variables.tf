variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy resources in"
  type        = string
  default     = "europe-west4"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "diabetic-fastapi"
}


variable "container_image" {
  description = "Container image for Cloud Run deployment"
  type        = string
}

variable "container_image_tag" {
  description = "Container image tag for versioning (e.g., 'latest', 'v1.0.0')"
  type        = string
  default     = "latest"
}
variable "log_level" {
  description = "Logging level for the FastAPI app (e.g., 'info', 'debug')"
  type        = string
  default     = "info"
}

variable "service_account_name" {
  description = "Service account name for Cloud Run"
  type        = string
  default     = "fastapi-sa"
}

# Frontend (Streamlit) variables
variable "frontend_service_name" {
  description = "Cloud Run service name for the frontend (Streamlit) app"
  type        = string
  default     = "diabetic-frontend"
}

variable "frontend_container_image" {
  description = "Container image for frontend Cloud Run deployment"
  type        = string
}

variable "frontend_container_image_tag" {
  description = "Container image tag for frontend versioning (e.g., 'latest', 'v1.0.0')"
  type        = string
  default     = "latest"
}

variable "frontend_service_account_name" {
  description = "Service account name for frontend Cloud Run"
  type        = string
  default     = "frontend-sa"
}


variable "artifacts_gcs_uri" {
  description = "GCS URI to the root directory containing all model and feature set artifacts"
  type        = string
}


variable "data_storage_bucket_name" {
  description = "Name of the data bucket that contains the raw data, the processed data and the data enriched with user inputs"
  type        = string
}
