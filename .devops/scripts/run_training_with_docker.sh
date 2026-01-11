# Build a Docker image using train.dockerfile
docker build -f dockerfiles/train.dockerfile . -t train:latest

# Run a container from the built image, and delete it after it exits
docker run --name experiment1 --rm train:latest 