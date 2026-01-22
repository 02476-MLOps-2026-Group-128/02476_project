# Diabetes classification

## Development
**Syncing dependencies from pyproject.toml:**
```bash
uv sync
```

**Installing pre-commit config:**
```bash
uv run pre-commit install
```


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

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
* [P] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [P] Load test your application (M24)
* [M] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [V] Create a frontend for your API (M26)

### Week 3

* [B] Check how robust your model is towards data drifting (M27)
* [P] Setup collection of input-output data from your deployed application (M27)
* [B] Deploy to the cloud a drift detection API (M27)
* [P] Instrument your API with a couple of system metrics (M28)
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

For the hyperparameter configuration we use Hydra config files, which can be found in the 'configs/hydra' directory. These parameters are added to the generated model folder, in a 'hydra' subfolder.

## API
We provide a FastAPI application for model inference. To run the API locally, run the following commands (in different terminals):

```bash
make run-api
```
```bash
uv run streamlit run frontend.py
```

The API expects the models to be stored in the `models/api_models/` directory, following a specific structure based on problem type, model type, feature set, and version. Each version folder should contain a `config.json` file specifying the model path and architecture details, along with the actual model file (e.g., `model.pt`).

A model is already given in the repository for testing purposes. You can find it at:
`models/api_models/diagnosed_diabetes/MLP/feature_set1/v1/`

To convert a Lightning checkpoint (`.ckpt`) to the required PyTorch state dictionary format (`.pt`), you can use the provided utility script:
```bash
uv run tools/ckpt_to_pt.py path/to/your/model.ckpt models/api_models/diagnosed_diabetes/MLP/feature_set1/v1/ --output-name model.pt
```
Make sure to create the corresponding `config.json` in the same directory to match your model's architecture.

### Data Drift
Data Drift (the change in data distribution between training and inference overtime) is monitored using the `Evidently` framework.

For saving user inputs, the API makes use of FastAPI's `BackgroundTasks` on each request to the `predict` endpoint to store the incoming data, making sure that the request is not delayed by the storage operation.
The location of where the data is stored depends on whether the API is running locally or in the cloud, which is determined by searching for the `AIP_STORAGE_URI` environment variable.

The predict endpoint receives a JSON payload containing both the raw feature input (for saving to disk) and processed (normalized) feature input (for model inference).

#### Locally
Locally, the incoming data is stored in the `data/data_drift/input_database.csv` file.
Locally, the API exposes a monitoring endpoint (`/monitoring`), which provides reports generated by Evidently about the data drift, data quality and target drift.
When the `/monitoring` endpoint is called, it uses the stored data and the training data to generate the reports.


#### In the cloud
For our cloud deployment, the incoming data is stored in a GCP Bucket (`diabetes-monitoring`).
Instead of expanding the API with a monitoring endpoint, we have created a dedicated reports API.
On start-up, this API reads the `processed_train_data.csv` from the GCP bucket.
When the `/report` endpoint is called, is uses the stored training data and fetches the saved user input data from the GCP bucket to generate the reports.
By default, the last 5 prediction inputs are used to generate the reports, but this can be changed by providing an `n` query parameter in the request.