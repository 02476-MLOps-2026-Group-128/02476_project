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


@pytest.mark.parametrize(
    ("optimizer_name", "expected_cls", "overrides"),
    [
        ("adam", torch.optim.Adam, ["optimizer=adam"]),
        ("sgd", torch.optim.SGD, ["optimizer=sgd"]),
    ],
)
def test_configure_optimizers_respects_config(
    cfg_factory,
    optimizer_name: str,
    expected_cls: type[torch.optim.Optimizer],
    overrides: list[str],
) -> None:
    cfg = cfg_factory(overrides)

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=TEST_INPUT_DIM,
        output_dim=1,
    )
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, expected_cls)
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == pytest.approx(cfg.optimizer.lr)
    assert param_group["weight_decay"] == pytest.approx(cfg.optimizer.weight_decay)
    if optimizer_name == "sgd":
        assert param_group["momentum"] == pytest.approx(cfg.optimizer.momentum)
