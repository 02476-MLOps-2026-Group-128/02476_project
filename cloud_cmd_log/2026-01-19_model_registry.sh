REGION=europe-west4
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME=diabetic-classification-model-bucket
CONTAINER_REGISTRY=europe-west4-docker.pkg.dev/$PROJECT_ID/container-registry

# Create the inference bucket
gcloud storage buckets create gs://$BUCKET_NAME/ \
    --location=$REGION \
    --uniform-bucket-level-access

# Upload a local api models to the bucket
gcloud storage cp -r ./models/api_models/ gs://$BUCKET_NAME/models/

# Upload the entire feature_sets directory
gcloud storage cp -r configs/feature_sets/ gs://$BUCKET_NAME/configs/
