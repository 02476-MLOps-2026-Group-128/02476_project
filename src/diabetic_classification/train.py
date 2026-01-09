from diabetic_classification.data import MyDataset
from diabetic_classification.model import Model


def train():
    """Train the model."""
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here


if __name__ == "__main__":
    train()
