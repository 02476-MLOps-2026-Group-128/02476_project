from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn


class TabularMLP(nn.Module):
    """A multilayer perceptron for tabular data.

    Args:
        input_dim: Number of input features.
        hidden_dims: Hidden layer sizes for the MLP.
        dropout: Dropout probability for hidden layers.
        output_dim: Number of output units (targets).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor with shape (batch, features).

        Returns:
            Model logits with shape (batch, output_dim).
        """
        return self.net(x)


class DiabetesClassifier(LightningModule):
    """LightningModule wrapper for the tabular MLP.

    Args:
        input_dim: Number of input features.
        lr: Learning rate.
        weight_decay: Weight decay for the optimizer.
        output_dim: Number of output units (targets).
        hidden_dims: Hidden layer sizes for the MLP.
        dropout: Dropout probability for hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dim: int = 1,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TabularMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
        )
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
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
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
