from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import typer
from diabetic_classification.data import DiabetesHealthDataset
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import AnyNode, DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig

from src.diabetic_classification.model import DiabetesClassifier

allowed_types = [DictConfig, ContainerMetadata, Any, dict, defaultdict, AnyNode, Metadata, ListConfig, list, int]


def export_to_onnx(
    checkpoint_path: str = typer.Argument(help="Path to .ckpt file"),
    output_path: str | None = typer.Argument(default=None, help="Path to save the ONNX model"),
) -> None:
    """Export the DiabetesClassifier model to ONNX format."""
    with initialize(version_base=None, config_path="../configs/hydra"):
        cfg = compose(config_name="config", return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    torch.serialization.add_safe_globals(allowed_types)

    model = DiabetesClassifier.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()

    data = DiabetesHealthDataset(
        data_dir=Path(cfg.data.data_dir),
        batch_size=1,
        feature_attributes=cfg.data.feature_attributes,
        target_attributes=cfg.data.target_attributes,
    )
    data.setup("fit")

    example_input = data.train_dataset.features[:1]

    if output_path is None:
        output_path = Path("onnx_models") / Path(Path(checkpoint_path).stem).with_suffix(".onnx")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.model,
        example_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    typer.run(export_to_onnx)
