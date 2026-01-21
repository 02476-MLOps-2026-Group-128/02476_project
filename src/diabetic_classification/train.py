from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier


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


@hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
def run_training(cfg) -> str:
    """Train the diabetes classifier with PyTorch Lightning, using the specified Hydra configuration."""
    pl.seed_everything(cfg.trainer.seed, workers=True)

    if not cfg.data.target_attributes:
        raise ValueError("Specify at least one target attribute.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(cfg.trainer.models_dir) / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    data = DiabetesHealthDataset(
        data_dir=Path(cfg.data.data_dir),
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
        val_split=cfg.trainer.val_split,
        feature_attributes=cfg.data.feature_attributes,
        target_attributes=cfg.data.target_attributes,
    )
    data.setup("fit")

    if data.target_columns is None:
        raise RuntimeError("Target columns were not resolved during dataset setup.")
    if data.train_dataset is None:
        raise RuntimeError("Training dataset was not prepared during setup.")

    pos_weight_tensor: torch.Tensor | None = None
    if cfg.trainer.use_pos_weight:
        try:
            pos_weight_tensor = compute_pos_weight(data.train_dataset.labels)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=data.train_dataset.features.shape[1],
        output_dim=len(data.target_columns),
        pos_weight=pos_weight_tensor,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=model_dir,
        deterministic=cfg.trainer.deterministic,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                save_top_k=cfg.trainer.model_checkpoint.save_top_k,
                monitor=cfg.trainer.model_checkpoint.monitor,
                mode=cfg.trainer.model_checkpoint.mode,
                save_last=cfg.trainer.model_checkpoint.save_last,
            )
        ],
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    checkpoint_callback = trainer.checkpoint_callback
    if isinstance(checkpoint_callback, ModelCheckpoint):
        final_ckpt = checkpoint_callback.best_model_path
    else:
        raise ValueError("ModelCheckpoint callback was not found or initialized.")
    return final_ckpt


@hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
def train(cfg) -> None:
    """
    Wrap the run_training function for Hydra.

    The wrapped function is detached for easier testing.
    """
    run_training(cfg)


if __name__ == "__main__":
    train()
