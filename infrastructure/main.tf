resource "google_storage_bucket" "model_registry" {
  name     = "diabetic-classification-model-bucket"
  location = var.region
  uniform_bucket_level_access = true
}

resource "google_service_account" "fastapi_sa" {
  account_id   = var.service_account_name
  display_name = "FastAPI Cloud Run"
  description  = "Service account for FastAPI Cloud Run deployment"
}

resource "google_project_iam_member" "fastapi_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.fastapi_sa.email}"
}

resource "google_cloud_run_service" "fastapi" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.fastapi_sa.email
      containers {
        image = "${var.container_image}:${var.container_image_tag}"
        env {
            name  = "ARTIFACTS_GCS_URI"
            value = var.artifacts_gcs_uri
        }
        env {
            name  = "DATA_STORAGE_BUCKET_NAME"
            value = var.data_storage_bucket_name
        }
        env {
            name  = "LOG_LEVEL"
            value = var.log_level
          }
        ports {
          name           = "http1"
          container_port = 8000
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "2"
        "run.googleapis.com/concurrency"   = "80"
        "run.googleapis.com/healthcheck-path" = "/"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true
}

resource "google_cloud_run_service_iam_member" "public_invoker" {
  location        = google_cloud_run_service.fastapi.location
  project         = var.project_id
  service         = google_cloud_run_service.fastapi.name
  role            = "roles/run.invoker"
  member          = "allUsers"
}

# Frontend (Streamlit) Cloud Run deployment
resource "google_service_account" "frontend_sa" {
  account_id   = var.frontend_service_account_name
  display_name = "Frontend Cloud Run"
  description  = "Service account for Frontend (Streamlit) Cloud Run deployment"
}

resource "google_cloud_run_service" "frontend" {
  name     = var.frontend_service_name
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.frontend_sa.email
      containers {
        image = "${var.frontend_container_image}:${var.frontend_container_image_tag}"
        env {
          name  = "BACKEND_URL"
          value = google_cloud_run_service.fastapi.status[0].url
        }
        ports {
          name           = "http1"
          container_port = 8080
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "2"
        "run.googleapis.com/concurrency"   = "80"
        "run.googleapis.com/healthcheck-path" = "/"
      }
    }
  }
  traffic {
    percent         = 100
    latest_revision = true
  }
  autogenerate_revision_name = true
}

resource "google_cloud_run_service_iam_member" "frontend_public_invoker" {
  location        = google_cloud_run_service.frontend.location
  project         = var.project_id
  service         = google_cloud_run_service.frontend.name
  role            = "roles/run.invoker"
  member          = "allUsers"
}
