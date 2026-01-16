from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import ModelCheckpoint

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def csv_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated string into a list of strings."""
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise typer.BadParameter("Provide at least one non-blank value.")
    return items


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute positive-class weights for binary targets.

    Args:
    ----
        labels: Tensor of shape (num_samples, num_targets) or (num_samples,).

    Returns:
    -------
        A tensor of shape (num_targets,) with positive-class weights.

    Raises:
    ------
        ValueError: If the targets are not binary or if any class is missing.
    """
    tensor = labels.detach().float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    unique_vals = torch.unique(tensor)
    if not torch.all((unique_vals == 0.0) | (unique_vals == 1.0)):
        raise ValueError("Positive-class weighting requires binary target(s).")
    positives = (tensor > 0.5).sum(dim=0)
    negatives = tensor.shape[0] - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Both classes must be present to compute pos_weight.")
    return negatives / positives


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    target_attributes: str = typer.Option(
        "diagnosed_diabetes",
        "--targets",
        metavar="a,b,c",
        help="Comma-separated target attribute name(s).",
    ),
    feature_attributes: str | None = typer.Option(
        None,
        "--feature-attributes",
        metavar="a,b,c",
        help="Comma-separated feature attributes.",
    ),
    feature_config: Path | None = typer.Option(
        None,
        "--feature-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a JSON file listing feature attributes.",
    ),
    exclude_feature_attributes: str | None = typer.Option(
        None,
        "--exclude-feature-attributes",
        metavar="a,b,c",
        help="Comma-separated features to remove.",
    ),
    batch_size: int = 256,
    max_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
    models_dir: Path = Path("models"),
    use_pos_weight: bool = typer.Option(
        True,
        "--use-pos-weight",
        help="Compute positive-class weights from the training targets for BCE loss.",
    ),
) -> None:
    """
    Train the diabetes classifier with PyTorch Lightning.

    Args:
    ----
        data_dir: Base data directory (contains raw/processed subfolders).
        target_attributes: Target attribute name(s), comma-separated for multiple.
        feature_attributes: Feature attribute name(s), comma-separated for multiple.
        feature_config: Path to JSON file with an explicit list of feature attributes.
        exclude_feature_attributes: Feature attribute name(s) to exclude, comma-separated for multiple.
        batch_size: Batch size for training and evaluation.
        max_epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for the optimizer.
        num_workers: DataLoader worker count.
        val_split: Fraction of training data reserved for validation.
        seed: Random seed for reproducibility.
        models_dir: Directory to save trained models.
        use_pos_weight: Whether to compute positive-class weights from the training targets.
    """
    pl.seed_everything(seed, workers=True)

    parsed_targets = csv_list(target_attributes)
    if not parsed_targets:
        raise typer.BadParameter("Specify at least one target attribute.")
    normalized_targets: list[str] | str = parsed_targets if len(parsed_targets) > 1 \
        else parsed_targets[0]

    parsed_features = csv_list(feature_attributes)
    if feature_config is not None:
        if parsed_features is not None:
            raise typer.BadParameter(
                "Use either --feature-attributes or --feature-config, not both.")
        try:
            payload = json.loads(feature_config.read_text())
        except OSError as exc:
            raise typer.BadParameter(
                f"Unable to read feature config: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(
                f"Feature config must be valid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise typer.BadParameter(
                "Feature config must be a JSON list of feature names.")
        parsed_features = []
        for attribute in payload:
            if not isinstance(attribute, str) or not attribute.strip():
                raise typer.BadParameter(
                    "Feature config entries must be non-empty strings.")
            parsed_features.append(attribute.strip())
        if not parsed_features:
            raise typer.BadParameter(
                "Feature config must describe at least one feature.")

    parsed_excludes = csv_list(exclude_feature_attributes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    data = DiabetesHealthDataset(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        target_attributes=normalized_targets,
        feature_attributes=parsed_features,
        exclude_feature_attributes=parsed_excludes,
    )
    data.setup("fit")

    if data.target_columns is None:
        raise RuntimeError(
            "Target columns were not resolved during dataset setup.")
    if data.train_dataset is None:
        raise RuntimeError("Training dataset was not prepared during setup.")

    pos_weight_tensor: torch.Tensor | None = None
    if use_pos_weight:
        try:
            pos_weight_tensor = compute_pos_weight(data.train_dataset.labels)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    model = DiabetesClassifier(
        input_dim=data.train_dataset.features.shape[1],
        lr=lr,
        weight_decay=weight_decay,
        output_dim=len(data.target_columns),
        pos_weight=pos_weight_tensor,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=models_dir,
        deterministic=True,
        log_every_n_steps=50,
        accelerator="auto",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                save_top_k=2,
                monitor="val_loss",
                mode="min",
                save_last=True,
            )
        ],
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    typer.run(train)
