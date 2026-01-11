from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import typer

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _parse_attributes(value: str | None) -> list[str] | str | None:
    """Parse comma-separated attribute names into a list or string.

    Args:
        value: Comma-separated attribute names or None.

    Returns:
        Parsed attribute list, a single attribute name, or None.
    """
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Attribute list cannot be empty.")
    if len(items) == 1:
        return items[0]
    return items


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    target_attributes: str = "diagnosed_diabetes",
    feature_attributes: str | None = None,
    exclude_feature_attributes: str | None = typer.Option(
        None, "--exclude-feature-attributes", "--exclude-attributes"
    ),
    batch_size: int = 256,
    max_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
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

    parsed_targets = _parse_attributes(target_attributes)
    parsed_features = _parse_attributes(feature_attributes)
    parsed_excludes = _parse_attributes(exclude_feature_attributes)
    
    
    data = DiabetesHealthDataset(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        target_attributes=parsed_targets,
        feature_attributes=parsed_features,
        exclude_feature_attributes=parsed_excludes,
    )
    data.setup("fit")

    if data.target_columns is None:
        raise RuntimeError("Target columns were not resolved during dataset setup.")

    model = DiabetesClassifier(
        lr=lr,
        weight_decay=weight_decay,
        output_dim=len(data.target_columns),
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        deterministic=True,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    typer.run(train)
