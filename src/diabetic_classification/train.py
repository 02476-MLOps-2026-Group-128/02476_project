from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from diabetic_classification.data import DiabetesHealthDataset
from diabetic_classification.model import DiabetesClassifier


@hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
def train(cfg) -> None:
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

    model = DiabetesClassifier(
        cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        input_dim=data.train_dataset.features.shape[1],
        output_dim=len(data.target_columns),
    )

    wandb_logger = WandbLogger(
        project="Diatech",
        entity="vojtadeconinck-danmarks-tekniske-universitet-dtu",
        name="diabetes-mlp",
        log_model=True,
    )
    wandb_logger.experiment.config.update(
        {
            "lr": cfg.optimizer.lr,
            "batch_size": cfg.trainer.batch_size,
            "max_epochs": cfg.trainer.max_epochs,
            "target_attributes": cfg.data.target_attributes,
            "feature_attributes": cfg.data.feature_attributes,
        }
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=model_dir,
        deterministic=cfg.trainer.deterministic,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
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
