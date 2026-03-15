"""Training configuration dataclass and TOML loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for contrastive text-audio training.

    All values have sensible defaults matching the project proposal.
    Load overrides from a TOML file with ``load_config()``.
    """

    # Model architecture
    embed_dim: int = 256
    projection_hidden_dim: int = 512
    temperature_init: float = 0.07

    # Encoder names
    text_encoder_name: str = "all-MiniLM-L6-v2"
    audio_encoder_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_epochs: int = 20
    eval_interval: int = 5
    grad_accumulation_steps: int = 1

    # Audio preprocessing
    n_mels: int = 128
    sample_rate: int = 16000
    max_audio_seconds: float = 10.0

    # Paths
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    log_dir: str = "runs"

    # Device: "auto", "cuda", or "cpu" - "auto" allowed in file
    # but not in-memory
    device: Literal["cuda", "cpu"] = "cuda"


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from a TOML file.

    Only keys that match TrainConfig fields are used;
    unknown keys are silently ignored.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Flatten: support a [train] section or top-level keys
    values = raw.get("train", raw)

    valid_fields = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in values.items() if k in valid_fields}

    if filtered.get("device", "auto") == "auto":
        filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(**filtered)
