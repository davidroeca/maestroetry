"""Training loop for contrastive text-audio model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch.optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from maestroetry.dataset import AudioTextDataset
from maestroetry.encoders import load_audio_encoder, load_text_encoder
from maestroetry.loss import info_nce_loss
from maestroetry.projection import ContrastiveModel, get_trainable_params

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from maestroetry.config import TrainConfig


def train_one_epoch(
    model: ContrastiveModel,
    dataloader: DataLoader[tuple[Tensor, str]],
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
    model.train()
    losses: list[float] = []
    for spectrograms, texts in dataloader:
        spectrograms = spectrograms.to(device)
        optimizer.zero_grad()
        text_embeds, audio_embeds, temperature = model(texts, spectrograms)
        loss = info_nce_loss(text_embeds, audio_embeds, temperature)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    if losses:
        return sum(losses, start=0) / len(losses)
    else:
        return 0.0


def train(config: TrainConfig) -> None:
    """Full training entrypoint.

    Loads encoders, builds the ContrastiveModel, sets up data
    loading, optimizer, scheduler, and TensorBoard writer, then
    runs the training loop for ``config.num_epochs``.

    Args:
        config: Training configuration.
    """
    text_encoder, text_embed_dim = load_text_encoder(
        name=config.text_encoder_name,
        device=config.device,
    )
    audio_encoder, audio_extractor, audio_embed_dim = load_audio_encoder(
        name=config.audio_encoder_name,
        device=config.device,
    )
    model = ContrastiveModel(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        audio_extractor=audio_extractor,
        text_embed_dim=text_embed_dim,
        audio_embed_dim=audio_embed_dim,
        projection_hidden=config.projection_hidden_dim,
        projection_out=config.embed_dim,
        temperature_init=config.temperature_init,
    )
    model.to(device=config.device)
    dataset = AudioTextDataset(
        manifest_path=Path(config.data_dir) / "manifest.csv",
        cache_dir=config.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / config.warmup_steps,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    writer = SummaryWriter(config.log_dir)
    for epoch in range(config.num_epochs):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
        )
        writer.add_scalar(
            "loss/train",
            loss,
            epoch,
        )
    writer.close()
