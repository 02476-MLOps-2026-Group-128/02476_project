import pytest
import torch

from diabetic_classification.model import DiabetesClassifier


TEST_INPUT_DIM = 5


@pytest.mark.parametrize(
    "batch_size, output_dim",
    [
        (1, 1),
        (8, 3),
    ],
)
def test_model_output_shape(cfg, batch_size: int, output_dim: int) -> None:

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=TEST_INPUT_DIM,
        output_dim=output_dim,
    )
    x = torch.randn(batch_size, TEST_INPUT_DIM)
    y = model(x)
    assert y.shape == (batch_size, output_dim), "Model output shape should match batch and output_dim."


def test_shared_step_raises_on_shape_mismatch(cfg) -> None:

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=TEST_INPUT_DIM,
        output_dim=1,
    )
    model.log = lambda *args, **kwargs: None
    x = torch.randn(2, TEST_INPUT_DIM)
    y = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="Logits shape"):
        model._shared_step((x, y), batch_idx=0, stage="train")
