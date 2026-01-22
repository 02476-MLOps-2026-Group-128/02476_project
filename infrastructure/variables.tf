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


variable "artifacts_gcs_uri" {
  description = "GCS URI to the root directory containing all model and feature set artifacts"
  type        = string
}
