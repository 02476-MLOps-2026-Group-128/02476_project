import pytest
from hydra import initialize, compose

import diabetic_classification.train as train_module

def test_valid_optimizer():
    with initialize(version_base=None, config_path="../configs/hydra"):
        cfg = compose(
            config_name="config",
            overrides=[
                "trainer.max_epochs=1",
                "paths.project_root=./",
                "data.data_dir=./data",
            ] # Use default valid optimizer
        )

        # Should not raise
        train_module.train_impl(cfg)

def test_invalid_optimizer():
    with initialize(version_base=None, config_path="../configs/hydra"):
        cfg = compose(
            config_name="config",
            overrides=[
                "optimizer.name=invalid",
                "trainer.max_epochs=1",
                "paths.project_root=./",
                "data.data_dir=./data",
            ]
        )

        # Capture exception
        with pytest.raises(ValueError) as exc_info:
            train_module.train_impl(cfg)

        # Print the actual exception message
        print("Exception message:", exc_info.value)

        # Assert the message manually
        assert "Unsupported optimizer" in str(exc_info.value)
