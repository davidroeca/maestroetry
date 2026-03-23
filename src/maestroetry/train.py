"""Training loop for contrastive text-audio model."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from maestroetry.dataset import AudioTextDataset
from maestroetry.encoders import (
    load_audio_encoder,
    load_text_encoder,
    unfreeze_audio_top_layers,
)
from maestroetry.evaluate import recall_at_k
from maestroetry.loss import info_nce_loss
from maestroetry.projection import ContrastiveModel, get_trainable_params

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from maestroetry.config import TrainConfig

logger = logging.getLogger(__name__)


def _eval_recall(
    model: ContrastiveModel,
    dataloader: DataLoader[tuple[Tensor, str]],
    device: str,
) -> dict[str, float]:
    """Compute Recall@k metrics over a full dataloader pass."""
    model.eval()
    all_text: list[Tensor] = []
    all_audio: list[Tensor] = []
    with torch.inference_mode():
        for spectrograms, texts in dataloader:
            spectrograms = spectrograms.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                text_embeds, audio_embeds, _ = model(texts, spectrograms)
            all_text.append(text_embeds.cpu())
            all_audio.append(audio_embeds.cpu())
    return recall_at_k(torch.cat(all_text), torch.cat(all_audio))


def _clip_and_step(
    model: ContrastiveModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    max_grad_norm: float,
) -> None:
    """Clip gradients (if enabled), step optimizer and scheduler."""
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _cosine_warmup_lambda(
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """Return an LR multiplier function for warmup + cosine decay.

    Applies as a multiplier to each optimizer param group's base LR,
    so differential learning rates are correctly scaled throughout.

    Args:
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of optimizer steps.

    Returns:
        A callable ``step -> float`` suitable for ``LambdaLR``.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def train_one_epoch(
    model: ContrastiveModel,
    dataloader: DataLoader[tuple[Tensor, str]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str = "cuda",
    grad_accumulation_steps: int = 1,
    max_grad_norm: float = 0.0,
    frozen_audio_layers: list[torch.nn.Module] | None = None,
) -> float:
    """Run one training epoch.

    For each batch: encode text and audio through the model,
    compute symmetric InfoNCE loss, backpropagate through
    trainable parameters, and step the optimizer every
    ``grad_accumulation_steps`` batches.

    Args:
        model: The ContrastiveModel to train.
        dataloader: Yields ``(spectrograms, texts)`` batches.
        optimizer: AdamW optimizer over trainable params.
        scheduler: Learning rate scheduler.
        device: Device to run on.
        grad_accumulation_steps: Number of batches to accumulate
            gradients over before each optimizer step.
        max_grad_norm: Maximum gradient norm for clipping. 0.0
            disables clipping.
        frozen_audio_layers: AST layers that should stay in eval
            mode even when the model is in train mode.

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    # Keep frozen AST layers in eval mode to preserve BatchNorm
    # behavior when only the top layers are unfrozen.
    if frozen_audio_layers:
        for layer in frozen_audio_layers:
            layer.eval()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for i, (spectrograms, texts) in enumerate(dataloader, 1):
        spectrograms = spectrograms.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            text_embeds, audio_embeds, temperature = model(texts, spectrograms)
            loss = info_nce_loss(text_embeds, audio_embeds, temperature)
            loss = loss / grad_accumulation_steps
        loss.backward()
        if i % grad_accumulation_steps == 0:
            _clip_and_step(model, optimizer, scheduler, max_grad_norm)
        losses.append(loss.item() * grad_accumulation_steps)
        if i % 20 == 0:
            logger.info("  batch %d - loss: %.4f", i, losses[-1])
    # Handle leftover batches that didn't complete a full
    # accumulation cycle.
    if len(dataloader) % grad_accumulation_steps != 0:
        _clip_and_step(model, optimizer, scheduler, max_grad_norm)
    if losses:
        return sum(losses, start=0) / len(losses)
    else:
        return 0.0


def _trainable_state_dict(model: ContrastiveModel) -> dict[str, torch.Tensor]:
    """Return state dict containing only trainable parameters."""
    trainable_keys = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    return {k: v for k, v in model.state_dict().items() if k in trainable_keys}


def save_checkpoint(
    path: Path,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
) -> None:
    """Save model checkpoint to disk.

    Only trainable parameters (projection heads and temperature)
    are saved; frozen encoder weights are excluded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": _trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    model_weights_only: bool = False,
) -> int:
    """Load checkpoint and return the next epoch to train.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state into.
        scheduler: Scheduler to restore state into.
        model_weights_only: If True, load only model weights and
            skip optimizer/scheduler state. Useful when resuming
            with changed hyperparameters.

    Returns:
        The next epoch to train from.
    """
    ckpt = torch.load(path, weights_only=False)
    # Filter checkpoint to only load currently trainable params,
    # so frozen encoder layers aren't overwritten with stale
    # fine-tuned weights from a different unfreeze config.
    trainable_keys = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    filtered = {k: v for k, v in ckpt["model"].items() if k in trainable_keys}
    skipped = set(ckpt["model"]) - trainable_keys
    if skipped:
        logger.info(
            "Skipped %d checkpoint keys not trainable in current config",
            len(skipped),
        )
    model.load_state_dict(filtered, strict=False)
    if model_weights_only:
        logger.info("Loaded model weights only (skipping optimizer/scheduler)")
        return 0
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1


def train(
    config: TrainConfig,
    resume: str | None = None,
    *,
    weights_only: bool = False,
) -> None:
    """Full training entrypoint.

    Loads encoders, builds the ContrastiveModel, sets up data
    loading, optimizer, scheduler, and TensorBoard writer, then
    runs the training loop for ``config.num_epochs``.

    Args:
        config: Training configuration.
        resume: Path to a checkpoint file to resume from.
        weights_only: If True, load only model weights from the
            checkpoint, ignoring optimizer/scheduler state.
    """
    torch.set_float32_matmul_precision("high")
    text_encoder, text_embed_dim = load_text_encoder(
        name=config.text_encoder_name,
        device=config.device,
    )
    audio_encoder, audio_extractor, audio_embed_dim = load_audio_encoder(
        name=config.audio_encoder_name,
        device=config.device,
    )
    finetune_audio = config.unfreeze_audio_layers > 0
    model = ContrastiveModel(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        audio_extractor=audio_extractor,
        text_embed_dim=text_embed_dim,
        audio_embed_dim=audio_embed_dim,
        projection_hidden=config.projection_hidden_dim,
        projection_out=config.embed_dim,
        projection_depth=config.projection_depth,
        projection_dropout=config.projection_dropout,
        temperature_init=config.temperature_init,
        finetune_audio=finetune_audio,
    )
    model.to(device=config.device)
    model.text_projection_head = torch.compile(model.text_projection_head)
    model.audio_projection_head = torch.compile(model.audio_projection_head)

    # Selectively unfreeze top AST layers if configured.
    frozen_audio_layers: list[torch.nn.Module] | None = None
    if finetune_audio:
        unfreeze_audio_top_layers(audio_encoder, config.unfreeze_audio_layers)
        total_layers = len(audio_encoder.encoder.layer)
        frozen_audio_layers = list(
            audio_encoder.encoder.layer[
                : total_layers - config.unfreeze_audio_layers
            ]
        )
        logger.info(
            "Unfreezing top %d AST layers (of %d total)",
            config.unfreeze_audio_layers,
            total_layers,
        )

    manifest_path = Path(config.data_dir) / "manifest.csv"
    train_dataset = AudioTextDataset(
        manifest_path=manifest_path,
        cache_dir=config.cache_dir,
        augment=config.spec_augment,
        spec_aug_freq_masks=config.spec_aug_freq_masks,
        spec_aug_freq_width=config.spec_aug_freq_width,
        spec_aug_time_masks=config.spec_aug_time_masks,
        spec_aug_time_width=config.spec_aug_time_width,
    )
    eval_dataset = AudioTextDataset(
        manifest_path=manifest_path,
        cache_dir=config.cache_dir,
        augment=False,
    )
    use_cuda = config.device != "cpu"
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )
    encoder_lr = config.encoder_learning_rate if finetune_audio else None
    optimizer = torch.optim.AdamW(
        get_trainable_params(model, encoder_lr=encoder_lr),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        steps_per_epoch / config.grad_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.num_epochs
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_cosine_warmup_lambda(
                warmup_steps=config.warmup_steps,
                total_steps=total_optimizer_steps,
            ),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / max(config.warmup_steps, 1),
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(
            Path(resume),
            model,
            optimizer,
            scheduler,
            model_weights_only=weights_only,
        )

    ckpt_dir = Path(config.checkpoint_dir)
    best_ckpt = ckpt_dir / "best.pt"
    writer = SummaryWriter(config.log_dir)
    effective_batch = config.batch_size * config.grad_accumulation_steps
    logger.info(
        "Training: %d epochs, %d samples, batch size %d"
        " (effective %d), schedule %s, device %s",
        config.num_epochs,
        len(train_dataset),
        config.batch_size,
        effective_batch,
        config.lr_schedule,
        config.device,
    )
    best_recall: float = -1.0
    for epoch in range(start_epoch, config.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, config.num_epochs)
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            grad_accumulation_steps=config.grad_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            frozen_audio_layers=frozen_audio_layers,
        )
        logger.info("  mean loss: %.4f", loss)
        writer.add_scalar("loss/train", loss, epoch)
        is_last = epoch == config.num_epochs - 1
        if is_last or (epoch + 1) % config.eval_interval == 0:
            metrics = _eval_recall(model, eval_dataloader, config.device)
            metrics_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info("  %s", metrics_str)
            for key, val in metrics.items():
                writer.add_scalar(f"metrics/{key}", val, epoch)
            recall = metrics.get("t2a_R@1", 0.0)
            if recall > best_recall:
                best_recall = recall
                save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch)
                logger.info(
                    "  best checkpoint updated (R@1=%.4f): %s",
                    best_recall,
                    best_ckpt,
                )
    logger.info(
        "Training complete. Best R@1=%.4f, checkpoint: %s",
        best_recall,
        best_ckpt,
    )
    writer.close()
