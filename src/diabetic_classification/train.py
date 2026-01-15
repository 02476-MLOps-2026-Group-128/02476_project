from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier


@hydra.main(
    version_base=None,
    config_path="../../configs/hydra",
    config_name="config"
)
def train(cfg) -> None:
    """Train the diabetes classifier with PyTorch Lightning, using the specified Hydra configuration."""
    pl.seed_everything(cfg.trainer.seed, workers=True)

    if not cfg.data.target_attributes:
        raise ValueError("Specify at least one target attribute.")

    targets = cfg.data.target_attributes
    normalized_targets: list[str] | str = targets if len(targets) > 1 \
        else targets[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(cfg.trainer.models_dir) / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    data = DiabetesHealthDataset(
        data_dir=Path(cfg.data.data_dir),
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
        val_split=cfg.trainer.val_split,
        target_attributes=normalized_targets,
        feature_attributes=cfg.data.feature_attributes,
        exclude_feature_attributes=cfg.data.exclude_feature_attributes,
    )
    data.setup("fit")

    if data.target_columns is None:
        raise RuntimeError("Target columns were not resolved during dataset setup.")
    if data.train_dataset is None:
        raise RuntimeError("Training dataset was not prepared during setup.")

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=data.train_dataset.features.shape[1],
        output_dim=len(data.target_columns),
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


if __name__ == "__main__":
    train()
