from diabetic_classification.model import Model
from diabetic_classification.data import DiabetesHealthDataset


def train():
    dataset = DiabetesHealthDataset("data/raw")
    model = Model()
    # add rest of your training code here
    print("PLACEHOLDER: the training started...")


if __name__ == "__main__":
    train()
