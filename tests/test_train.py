from types import SimpleNamespace

import pytest
import torch

import diabetic_classification.train as train_module


def test_valid_optimizer(cfg_factory) -> None:
    cfg = cfg_factory([
        "trainer.max_epochs=1",
    ])

    train_module.train_impl(cfg)


def test_invalid_optimizer(cfg_factory) -> None:
    cfg = cfg_factory([
        "optimizer.name=invalid",
        "trainer.max_epochs=1",
    ])

    # Capture exception
    with pytest.raises(ValueError) as exc_info:
        train_module.train_impl(cfg)

    # Print the actual exception message
    print("Exception message:", exc_info.value)

    # Assert the message manually
    assert "Unsupported optimizer" in str(exc_info.value)


def test_train_impl_runs_fit_and_test(cfg_factory, tmp_path, monkeypatch) -> None:
    class DummyDataModule:
        def __init__(self, *args, **kwargs) -> None:
            self.target_columns = None
            self.train_dataset = None

        def setup(self, stage: str | None = None) -> None:
            self.target_columns = ["diagnosed_diabetes"]
            self.train_dataset = SimpleNamespace(features=torch.zeros((2, 5)))

    class DummyWandbLogger:
        def __init__(self, *args, **kwargs) -> None:
            self.experiment = SimpleNamespace(config={})

    class DummyTrainer:
        last_instance = None

        def __init__(self, *args, **kwargs) -> None:
            DummyTrainer.last_instance = self
            self.calls = []
            self.fit_args = None
            self.test_args = None

        def fit(self, model, datamodule=None) -> None:
            self.calls.append("fit")
            self.fit_args = (model, datamodule)

        def test(self, model, datamodule=None) -> None:
            self.calls.append("test")
            self.test_args = (model, datamodule)

    monkeypatch.setattr(train_module, "DiabetesHealthDataset", DummyDataModule)
    monkeypatch.setattr(train_module, "WandbLogger", DummyWandbLogger)
    monkeypatch.setattr(train_module.pl, "Trainer", DummyTrainer)

    cfg = cfg_factory(
        [
            f"trainer.models_dir={tmp_path.as_posix()}",
            "trainer.max_epochs=1",
        ]
    )

    train_module.train_impl(cfg)

    trainer = DummyTrainer.last_instance
    assert trainer is not None
    assert trainer.calls == ["fit", "test"]
    assert trainer.fit_args is not None
    assert trainer.test_args is not None
    assert trainer.fit_args[1] is trainer.test_args[1]
    assert isinstance(trainer.fit_args[1], DummyDataModule)
