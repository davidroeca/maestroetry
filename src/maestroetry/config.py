"""Training configuration dataclass and TOML loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for CLAP-based contrastive fine-tuning.

    All values have sensible defaults. Override per-run via TOML
    files loaded with ``load_config()``.
    """

    # CLAP model + fine-tuning
    clap_model_name: str = "laion/larger_clap_music"
    unfreeze_audio_layers: int = 2
    unfreeze_text_layers: int = 2
    encoder_learning_rate: float = 1e-6
    temperature_init: float = 0.07

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    num_epochs: int = 100
    eval_interval: int = 5
    early_stopping_patience: int = 10
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Audio preprocessing
    max_audio_seconds: float = 10.0

    # Paths
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"

    # Device: "auto" allowed in file but normalized to cuda/cpu
    device: Literal["cuda", "cpu"] = "cuda"


def load_config(path: str | Path, **overrides: str) -> TrainConfig:
    """Load a TrainConfig from a TOML file.

    Only keys that match TrainConfig fields are used; unknown keys
    are silently ignored. Keyword arguments override file values.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Support a [train] section or top-level keys.
    values = raw.get("train", raw)

    valid_fields = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in values.items() if k in valid_fields}
    filtered.update({k: v for k, v in overrides.items() if k in valid_fields})

    if filtered.get("device", "auto") == "auto":
        filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(**filtered)
