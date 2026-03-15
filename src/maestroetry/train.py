"""Training loop for contrastive text-audio model."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from maestroetry.config import TrainConfig
    from maestroetry.projection import ContrastiveModel


def train_one_epoch(
    model: ContrastiveModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str = "cuda",
) -> float:
    """Run one training epoch.

    For each batch: encode text and audio through the model,
    compute symmetric InfoNCE loss, backpropagate through
    projection heads only, and step the optimizer.

    Args:
        model: The ContrastiveModel to train.
        dataloader: Yields ``(spectrograms, texts)`` batches.
        optimizer: AdamW optimizer over trainable params.
        scheduler: Learning rate scheduler (linear warmup).
        device: Device to run on.

    Returns:
        Mean loss over the epoch.
    """
    raise NotImplementedError


def train(config: TrainConfig) -> None:
    """Full training entrypoint.

    Loads encoders, builds the ContrastiveModel, sets up data
    loading, optimizer, scheduler, and TensorBoard writer, then
    runs the training loop for ``config.num_epochs``.

    Args:
        config: Training configuration.
    """
    raise NotImplementedError
