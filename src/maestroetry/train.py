"""Training loop for CLAP-based contrastive text-audio model."""

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

from maestroetry.dataset import (
    ClapFeaturesTextDataset,
    UniqueAudioBatchSampler,
    cache_clap_features,
    make_clap_collate_fn,
)
from maestroetry.encoders import (
    load_clap_model,
    unfreeze_clap_audio_top_layers,
    unfreeze_clap_text_top_layers,
)
from maestroetry.evaluate import recall_at_k
from maestroetry.loss import info_nce_loss
from maestroetry.model import ContrastiveModel, get_trainable_params

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from maestroetry.config import TrainConfig

logger = logging.getLogger(__name__)

_CLAP_SR = 48000


def _eval_recall(
    model: ContrastiveModel,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Compute Recall@k metrics over a full dataloader pass."""
    model.eval()
    all_text: list[Tensor] = []
    all_audio: list[Tensor] = []
    with torch.inference_mode():
        for audio_inputs, text_inputs in dataloader:
            text_embeds, audio_embeds, _ = model.forward_preprocessed(
                text_inputs, audio_inputs
            )
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
    """Return an LR multiplier function for warmup + cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def train_one_epoch(
    model: ContrastiveModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    grad_accumulation_steps: int = 1,
    max_grad_norm: float = 0.0,
) -> float:
    """Run one training epoch over the preprocessed-batch dataloader."""
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for i, (audio_inputs, text_inputs) in enumerate(dataloader, 1):
        text_embeds, audio_embeds, temperature = model.forward_preprocessed(
            text_inputs, audio_inputs
        )
        loss = info_nce_loss(text_embeds, audio_embeds, temperature)
        loss = loss / grad_accumulation_steps
        loss.backward()
        if i % grad_accumulation_steps == 0:
            _clip_and_step(model, optimizer, scheduler, max_grad_norm)
        losses.append(loss.item() * grad_accumulation_steps)
        if i % 20 == 0:
            logger.info("  batch %d - loss: %.4f", i, losses[-1])
    if len(dataloader) % grad_accumulation_steps != 0:
        _clip_and_step(model, optimizer, scheduler, max_grad_norm)
    if losses:
        return sum(losses, start=0) / len(losses)
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
    """Save trainable params + optimizer/scheduler state to disk."""
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
    """Load checkpoint and return the next epoch to train."""
    ckpt = torch.load(path, weights_only=False)
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
    """Full training entrypoint."""
    torch.set_float32_matmul_precision("high")

    clap, processor = load_clap_model(config.clap_model_name, config.device)
    model = ContrastiveModel(clap, processor, config.temperature_init)
    model.to(device=config.device)
    unfreeze_clap_audio_top_layers(clap, config.unfreeze_audio_layers)
    unfreeze_clap_text_top_layers(clap, config.unfreeze_text_layers)
    logger.info(
        "Unfrozen CLAP layers: audio=%d, text=%d",
        config.unfreeze_audio_layers,
        config.unfreeze_text_layers,
    )

    manifest_path = Path(config.data_dir) / "manifest.csv"
    audio_dir = Path(config.data_dir) / "audio"
    cache_clap_features(
        audio_dir=audio_dir,
        cache_dir=config.cache_dir,
        processor=processor,
        model_name=config.clap_model_name,
        sr=_CLAP_SR,
        max_seconds=config.max_audio_seconds,
    )

    train_dataset = ClapFeaturesTextDataset(
        manifest_path=manifest_path,
        cache_dir=config.cache_dir,
        model_name=config.clap_model_name,
        split="train",
    )
    eval_dataset = ClapFeaturesTextDataset(
        manifest_path=manifest_path,
        cache_dir=config.cache_dir,
        model_name=config.clap_model_name,
        split="eval",
    )
    use_cuda = config.device != "cpu"
    collate_fn = make_clap_collate_fn(processor)
    persistent = config.num_workers > 0
    dataloader = DataLoader(
        train_dataset,
        batch_sampler=UniqueAudioBatchSampler(
            train_dataset.track_ids,
            batch_size=config.batch_size,
        ),
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )
    optimizer = torch.optim.AdamW(
        get_trainable_params(
            model,
            encoder_lr=config.encoder_learning_rate,
            main_lr=config.learning_rate,
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        steps_per_epoch / config.grad_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_warmup_lambda(
            warmup_steps=config.warmup_steps,
            total_steps=total_optimizer_steps,
        ),
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
        "Training: %d epochs, %d samples, batch size %d (effective %d), device %s",
        config.num_epochs,
        len(train_dataset),
        config.batch_size,
        effective_batch,
        config.device,
    )
    best_recall: float = -1.0
    evals_without_improvement: int = 0
    patience = config.early_stopping_patience
    for epoch in range(start_epoch, config.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, config.num_epochs)
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accumulation_steps=config.grad_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
        )
        logger.info("  mean loss: %.4f", loss)
        writer.add_scalar("loss/train", loss, epoch)
        is_last = epoch == config.num_epochs - 1
        if is_last or (epoch + 1) % config.eval_interval == 0:
            metrics = _eval_recall(model, eval_dataloader)
            metrics_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info("  %s", metrics_str)
            for key, val in metrics.items():
                writer.add_scalar(f"metrics/{key}", val, epoch)
            recall = metrics.get("t2a_R@1", 0.0)
            if recall > best_recall:
                best_recall = recall
                evals_without_improvement = 0
                save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch)
                logger.info(
                    "  best checkpoint updated (R@1=%.4f): %s",
                    best_recall,
                    best_ckpt,
                )
            else:
                evals_without_improvement += 1
                if patience > 0 and evals_without_improvement >= patience:
                    logger.info(
                        "  early stopping: no improvement for %d eval cycles",
                        patience,
                    )
                    break
    logger.info(
        "Training complete. Best R@1=%.4f, checkpoint: %s",
        best_recall,
        best_ckpt,
    )
    writer.close()
