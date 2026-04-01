"""Evaluate best.pt checkpoints across all run directories and print R@1.

Still waiting on tensorboard 2.21, which addresses the issue of tensorboard
not working on python 3.13:

https://github.com/tensorflow/tensorboard/issues/7064

Issue closed but not yet released. Once released, this script should be deleted
and replaced with a tensorboard-focused implementation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maestroetry.config import TrainConfig, load_config
from maestroetry.dataset import AudioTextDataset
from maestroetry.encoders import (
    load_audio_encoder,
    load_text_encoder,
    unfreeze_audio_top_layers,
    unfreeze_text_top_layers,
)
from maestroetry.evaluate import recall_at_k
from maestroetry.projection import ContrastiveModel

logging.basicConfig(level=logging.WARNING)


def build_model(config: TrainConfig) -> ContrastiveModel:
    """Build a ContrastiveModel from config (no torch.compile)."""
    text_encoder, text_embed_dim = load_text_encoder(
        name=config.text_encoder_name,
        device=config.device,
    )
    audio_encoder, audio_extractor, audio_embed_dim = load_audio_encoder(
        name=config.audio_encoder_name,
        device=config.device,
    )
    finetune_audio = config.unfreeze_audio_layers > 0
    finetune_text = config.unfreeze_text_layers > 0
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
        finetune_text=finetune_text,
    )
    model.to(device=config.device)
    if finetune_audio:
        unfreeze_audio_top_layers(audio_encoder, config.unfreeze_audio_layers)
    if finetune_text:
        unfreeze_text_top_layers(text_encoder, config.unfreeze_text_layers)
    return model


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove '_orig_mod.' prefix that torch.compile adds to key names."""
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        cleaned[k.replace("._orig_mod", "")] = v
    return cleaned


def eval_checkpoint(
    ckpt_path: Path,
    config: TrainConfig,
    model: ContrastiveModel,
    eval_loader: DataLoader[tuple[torch.Tensor, str]],
) -> dict[str, float]:
    """Load a checkpoint into the model and compute recall metrics."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=config.device)
    cleaned = _strip_compile_prefix(ckpt["model"])
    model.load_state_dict(cleaned, strict=False)

    model.eval()
    all_text: list[torch.Tensor] = []
    all_audio: list[torch.Tensor] = []
    with torch.inference_mode():
        for spectrograms, texts in eval_loader:
            spectrograms = spectrograms.to(config.device, non_blocking=True)
            with torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16):
                text_embeds, audio_embeds, _ = model(texts, spectrograms)
            all_text.append(text_embeds.cpu())
            all_audio.append(audio_embeds.cpu())
    return recall_at_k(torch.cat(all_text), torch.cat(all_audio))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate R@1 for all run checkpoints")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Root directory containing run subdirectories (default: checkpoints)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Config file for model architecture (default: configs/default.toml)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory with manifest.csv (default: data)",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Specific run directories to evaluate (default: all)",
    )
    args = parser.parse_args()

    config = load_config(args.config, data_dir=args.data_dir)
    root = Path(args.checkpoint_root)

    # Find all best.pt files
    if args.runs:
        ckpt_paths = [(root / r / "best.pt") for r in args.runs]
    else:
        ckpt_paths = sorted(root.glob("*/best.pt"))
        # Also check for a top-level best.pt
        top_best = root / "best.pt"
        if top_best.exists():
            ckpt_paths.insert(0, top_best)

    if not ckpt_paths:
        print("No best.pt checkpoints found.")
        return

    # Build model and eval dataset once
    model = build_model(config)
    manifest_path = Path(config.data_dir) / "manifest.csv"
    eval_dataset = AudioTextDataset(
        manifest_path=manifest_path,
        cache_dir=config.cache_dir,
        split="eval",
        augment=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=config.device != "cpu",
    )

    print(f"{'Checkpoint':<40} {'Epoch':>6}  {'t2a_R@1':>8}  {'a2t_R@1':>8}  {'t2a_R@5':>8}  {'t2a_R@10':>9}")
    print("-" * 115)

    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            print(f"{str(ckpt_path):<40} {'N/A':>6}  (not found)")
            continue

        # Read epoch from checkpoint
        ckpt_data = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        epoch = ckpt_data.get("epoch", -1)

        metrics = eval_checkpoint(ckpt_path, config, model, eval_loader)
        label = ckpt_path.parent.name if ckpt_path.parent != root else "(root)"
        print(
            f"{label:<40} {epoch + 1:>6}  "
            f"{metrics.get('t2a_R@1', 0):>8.4f}  "
            f"{metrics.get('a2t_R@1', 0):>8.4f}  "
            f"{metrics.get('t2a_R@5', 0):>8.4f}  "
            f"{metrics.get('t2a_R@10', 0):>9.4f}"
        )


if __name__ == "__main__":
    main()
