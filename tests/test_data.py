from torch.utils.data import Dataset

from diabetic_classification.data import DiabetesHealthDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = DiabetesHealthDataset("data/raw")
    assert isinstance(dataset, Dataset)
