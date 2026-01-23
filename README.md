# Diabetes classification

## Development
**Installing pre-commit config:**
```bash
uv run pre-commit install
```



## Deployment

### Terraform Infrastructure

See `infrastructure/README.md` for detailed instructions on deploying and managing the GCP infrastructure (Cloud Run, GCS, IAM, Artifact Registry) with Terraform.

## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very point on the
checklist for the exam. The parenthesis at the end indicates what module the bullet point is related to.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [Everyone if time is available] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [P] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [B] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [V] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [M] Add a continues workflow that triggers when data changes (M19)
* [P] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [M] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [V] Create a frontend for your API (M26)

### Week 3

* [B] Check how robust your model is towards data drifting (M27)
* [M] Setup collection of input-output data from your deployed application (M27)
* [B] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [V] Setup cloud monitoring of your instrumented application (M28)
* [M] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [-] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [-] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [Everyone if time is available] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub


## Training
Run: uv run python -m diabetic_classification.train

The training script uses Hydra for hyperparameter configuration management. The configuration files are located at `configs/hydra`. The parameters used are logged together with the model, in a 'hydra' subfolder.

### Hyperparameter sweep
It is possible to overwrite the hydra configuration parameters from the command line, using the `-m` (`--multirun`) flag. Hydra will then run the training script multiple times, once for each combination of the specified parameters.

The multirun flag can generate a lot of training runs, therefore we use it with the parameters `trainer.enable_checkpointing=false trainer.logger=false`, such that no checkpoints and ligtning logs are saved, keeping only the Hydra logs in `/multirun/` folder (test accuracy and loss are logged to the `train.log` file).

There are a log of hyperparameters that could be tuned, such as:
- learning-rate
- batch-size
- optimizer
- dropout rate
- epochs
- hidden dimensions

Trying a sweep over all these parameters would quickly generate a lot of training runs. Therefore, we have opted for the following sweep:

```bash
uv run src/diabetic_classification/train.py -m trainer.enable_checkpointing=false trainer.logger=false \
optimizer.lr=0.001,0.0001 \
trainer.max_epochs=10,20 \
model.hidden_dims="[128,64]","[256,128]" \
model.dropout=0.1,0.3
```

This command generates 16 training runs. The results of the sweep did not indicate a significant difference in performance between the different hyperparameter combinations, with the best combination achieving a test accuracy of 81.935% (all accuracies were between in between 81% and 82%). The best combination seemed to be:
- learning-rate: 0.001
- epochs: 10
- hidden dimensions: [256,128]
- dropout: 0.1

These parameters are set as default in `configs/hydra/

### Remotely building the docker image
It is possible to build the training docker image remotely in GCP using Cloud Build. This is done automatically when new code is pushed to the main branch (`.github/workflows/tests.yaml`), but it can also be triggered manually by running the following command:

```bash
gcloud builds submit . --config=cloudbuild.yaml
```

The image can be pulled from the Artifact Registry in GCP using the following command:

```bash
docker pull europe-west4-docker.pkg.dev/diabetic-classification-484510/container-registry/train:latest
```

This requires that docker is authenticated with GCP:
```bash
gcloud auth configure-docker europe-west4-docker.pkg.dev
```

### Training in the cloud
The GCP project now contains the image of the latest training code (train.dockerfile). This image be used to run training in GCP using Vertex AI. Our GCP project has access to a GPU, so in `config_gpu.yaml` we specify that we want to use a GPU for training: `NVIDIA_TESLA_T4`. To create a training job in Vertex AI, use the following command:

```bash
gcloud ai custom-jobs create --region=europe-west4 --display-name=train --config=configs/config_gpu.yaml
```

This command uses the configuration file `configs/config_gpu.yaml`, which specifies where to find the docker image, the machine type to use (which we have configured to use a GPU), the output directory, and the Weights & Biases API key for logging.

The status of the training job can be monitored in the GCP console: [here](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=diabetic-classification-484510&vertex_ai_region=europe-west4).