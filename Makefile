# Runs the test suite
test:
	bash .devops/scripts/run_tests.sh

# Builds a Docker image and runs training inside a container
train-in-docker:
	bash .devops/scripts/run_training_with_docker.sh

# Runs the API server
run-api:
	bash .devops/scripts/run_api.sh

# Run the frontend server
run-frontend:
	bash .devops/scripts/run_frontend.sh

# Run cloudbuild
cloudbuild:
	bash .devops/scripts/run_cloudbuild.sh

# Update the data
update-data:
	bash .devops/scripts/run_update_data.sh
