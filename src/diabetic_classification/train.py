from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import typer

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise typer.BadParameter("Provide at least one non-blank value.")
    return items

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
) -> None:
    """Train the diabetes classifier with PyTorch Lightning.

    Args:
        data_dir: Base data directory (contains raw/processed subfolders).
        target_attributes: Target attribute name(s), comma-separated for multiple.
        feature_attributes: Feature attribute name(s), comma-separated for multiple.
        exclude_feature_attributes: Feature attribute name(s) to exclude, comma-separated for multiple.
        batch_size: Batch size for training and evaluation.
        max_epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for the optimizer.
        num_workers: DataLoader worker count.
        val_split: Fraction of training data reserved for validation.
        seed: Random seed for reproducibility.
    """
    pl.seed_everything(seed, workers=True)

    parsed_targets = csv_list(target_attributes)
    if not parsed_targets:
        raise typer.BadParameter("Specify at least one target attribute.")
    normalized_targets: list[str] | str = parsed_targets if len(parsed_targets) > 1 \
                                            else parsed_targets[0]

    parsed_features = csv_list(feature_attributes)
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
        raise RuntimeError("Target columns were not resolved during dataset setup.")
    if data.train_dataset is None:
        raise RuntimeError("Training dataset was not prepared during setup.")

    model = DiabetesClassifier(
        input_dim=data.train_dataset.features.shape[1],
        lr=lr,
        weight_decay=weight_decay,
        output_dim=len(data.target_columns),
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
