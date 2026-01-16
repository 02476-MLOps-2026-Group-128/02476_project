# Runs the test suite
test:
	bash .devops/scripts/run_tests.sh

# Builds a Docker image and runs training inside a container
train-in-docker:
	bash .devops/scripts/run_training_with_docker.sh

# Runs the API server
run-api:
	bash .devops/scripts/run_api.sh
