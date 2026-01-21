from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import typer
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import AnyNode, DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier

ALLOWED_TYPES: list[Callable[..., Any] | tuple[Callable[..., Any], str]] = [
    DictConfig,
    ContainerMetadata,
    dict,
    defaultdict,
    AnyNode,
    Metadata,
    ListConfig,
    list,
    int,
]

PROCESSED_INPUT_FEATURES = 38


def export_to_onnx(
    checkpoint_path: str,
    output: str | None = None,
) -> Path:
    """
    Export the DiabetesClassifier model to ONNX format.

    Arguments:
    ---------
        checkpoint_path (str): Path to the .ckpt file of the trained model.
        output (str | None): Path to save the ONNX model.
            If None, saves alongside the checkpoint with .onnx extension.

    Returns:
    -------
        Path: The path to the saved ONNX model.

    """
    with initialize(version_base=None, config_path="../configs/hydra"):
        cfg = compose(config_name="config", return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    torch.serialization.add_safe_globals(ALLOWED_TYPES)

    model = DiabetesClassifier.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()

    example_input = torch.randn(1, PROCESSED_INPUT_FEATURES)

    output_file = Path(output) if output is not None else Path(checkpoint_path).with_suffix(".onnx")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.model,
        (example_input,),
        output_file,
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

    return output_file


def run_export(
    checkpoint_path: str = typer.Argument(help="Path to .ckpt file"),
    output: str = typer.Argument(default=None, help="Path to save the ONNX model"),
) -> None:
    """Run the export_to_onnx function via Typer CLI."""
    export_to_onnx(checkpoint_path, output)


if __name__ == "__main__":
    typer.run(run_export)
