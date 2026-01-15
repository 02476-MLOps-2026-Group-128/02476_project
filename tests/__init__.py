from pathlib import Path

_TEST_ROOT = Path(__file__).parent  # root of test folder
_PROJECT_ROOT = _TEST_ROOT.parent  # root of project
_DATA_PATH = _PROJECT_ROOT / "data"  # root of data
_PROCESSED_DATA_PATH = _DATA_PATH / "processed"  # processed data folder
_RAW_DATA_PATH = _DATA_PATH / "raw"  # raw data folder
_MODELS_PATH = _PROJECT_ROOT / "models"  # root of models
