output "cloud_run_url" {
  description = "The URL of the deployed Cloud Run service."
  value       = google_cloud_run_service.fastapi.status[0].url
}

output "service_account_email" {
  description = "The email of the created service account."
  value       = google_service_account.fastapi_sa.email
}
