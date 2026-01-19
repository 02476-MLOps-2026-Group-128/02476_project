REGION=europe-west4
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME=diabetic-classification-model-bucket
CONTAINER_REGISTRY=europe-west4-docker.pkg.dev/$PROJECT_ID/container-registry

# Create the inference bucket
gcloud storage buckets create gs://$BUCKET_NAME/ \
    --location=$REGION \
    --uniform-bucket-level-access

# Upload a local model folder to the bucket
gcloud storage cp -r ./models/diagnosed_diabetes/MLP/feature_set1/v1/ gs://$BUCKET_NAME/models/diagnosed_diabetes/MLP/feature_set1/

# Upload the models.json to the root of your model folder
gcloud storage cp configs/models.json gs://$BUCKET_NAME/configs/models.json

# Upload the entire feature_sets directory
gcloud storage cp -r configs/feature_sets/ gs://$BUCKET_NAME/configs/
