from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn


class TabularCNN(nn.Module):
    """A lightweight 1D CNN for tabular data.

    Args:
        conv_channels: Output channels for the two convolutional layers.
        kernel_size: Kernel size for the convolutional layers.
        dropout: Dropout probability for the classifier head.
        output_dim: Number of output units (targets).
    """

    def __init__(
        self,
        conv_channels: tuple[int, int] = (16, 32),
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor with shape (batch, features).

        Returns:
            Model logits with shape (batch, output_dim).
        """
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


class DiabetesClassifier(LightningModule):
    """LightningModule wrapper for the tabular CNN.

    Args:
        lr: Learning rate.
        weight_decay: Weight decay for the optimizer.
        output_dim: Number of output units (targets).
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, output_dim: int = 1) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TabularCNN(output_dim=output_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward input through the model.

        Args:
            x: Input tensor with shape (batch, features).

        Returns:
            Logits with shape (batch, output_dim).
        """
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute loss and log metrics for a training stage.

        Args:
            batch: Tuple of (features, targets).
            stage: Stage name for logging.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        logits = self(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        if logits.shape != y.shape:
            raise ValueError(f"Logits shape {tuple(logits.shape)} does not match targets {tuple(y.shape)}.")
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        acc = (probs > 0.5).eq(y > 0.5).float().mean()
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning validation step."""
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning test step."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
