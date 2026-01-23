from types import SimpleNamespace

import pytest
import torch

import diabetic_classification.train as train_module

TRAIN_FN = getattr(train_module.train, "__wrapped__", train_module.train)


class DummyDataModule:
    """Minimal data module stub for training tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.target_columns: list[str] | None = None
        self.train_dataset: SimpleNamespace | None = None

    def setup(self, stage: str | None = None) -> None:
        self.target_columns = ["diagnosed_diabetes"]
        self.train_dataset = SimpleNamespace(
            features=torch.zeros((2, 5)),
            labels=torch.tensor([0.0, 1.0]),
        )


class DummyTrainer:
    """Trainer stub that records calls and validates optimizer setup."""

    last_instance: "DummyTrainer | None" = None

    def __init__(self, *args, **kwargs) -> None:
        DummyTrainer.last_instance = self
        self.calls: list[str] = []
        self.fit_args: tuple[object, object | None] | None = None
        self.test_args: tuple[object, object | None] | None = None

    def fit(self, model, datamodule=None) -> None:
        self.calls.append("fit")
        self.fit_args = (model, datamodule)
        model.configure_optimizers()

    def test(self, model, datamodule=None) -> None:
        self.calls.append("test")
        self.test_args = (model, datamodule)


def test_valid_optimizer(cfg_factory, monkeypatch, tmp_path) -> None:
    cfg = cfg_factory([
        f"trainer.models_dir={tmp_path.as_posix()}",
        "trainer.max_epochs=1",
    ])

    monkeypatch.setattr(train_module, "DiabetesHealthDataset", DummyDataModule)
    monkeypatch.setattr(train_module.pl, "Trainer", DummyTrainer)

    TRAIN_FN(cfg)


def test_invalid_optimizer(cfg_factory, monkeypatch, tmp_path) -> None:
    cfg = cfg_factory([
        "optimizer.name=invalid",
        f"trainer.models_dir={tmp_path.as_posix()}",
        "trainer.max_epochs=1",
    ])

    monkeypatch.setattr(train_module, "DiabetesHealthDataset", DummyDataModule)
    monkeypatch.setattr(train_module.pl, "Trainer", DummyTrainer)

    # Capture exception
    with pytest.raises(ValueError) as exc_info:
        TRAIN_FN(cfg)

    # Print the actual exception message
    print("Exception message:", exc_info.value)

    # Assert the message manually
    assert "Unsupported optimizer" in str(exc_info.value)


def test_train_runs_fit_and_test(cfg_factory, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "DiabetesHealthDataset", DummyDataModule)
    monkeypatch.setattr(train_module.pl, "Trainer", DummyTrainer)

    cfg = cfg_factory(
        [
            f"trainer.models_dir={tmp_path.as_posix()}",
            "trainer.max_epochs=1",
        ]
    )

    TRAIN_FN(cfg)

    trainer = DummyTrainer.last_instance
    assert trainer is not None
    assert trainer.calls == ["fit", "test"]
    assert trainer.fit_args is not None
    assert trainer.test_args is not None
    assert trainer.fit_args[1] is trainer.test_args[1]
    assert isinstance(trainer.fit_args[1], DummyDataModule)
