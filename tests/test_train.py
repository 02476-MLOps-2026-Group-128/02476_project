import pytest
from hydra import initialize, compose

import diabetic_classification.train as train_module

def test_valid_optimizer(cfg_factory):
    cfg = cfg_factory([
        "trainer.max_epochs=1",
    ])
    
    train_module.train_impl(cfg)

def test_invalid_optimizer(cfg_factory):
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
