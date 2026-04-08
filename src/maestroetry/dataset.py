"""Audio preprocessing and dataset for text-audio pairs."""

from __future__ import annotations

import csv
import logging
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from transformers import ClapProcessor

from maestroetry.encoders import (
    CLAP_AUDIO_INPUT_KEYS,
    CLAP_TEXT_INPUT_KEYS,
)

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}


def audio_to_waveform(
    path: str | Path,
    sr: int = 48000,
    max_seconds: float = 10.0,
) -> Tensor:
    """Load an audio file as a raw waveform tensor.

    Resamples to ``sr``, truncates or zero-pads to exactly
    ``max_seconds`` so all waveforms in a batch have equal length.

    Args:
        path: Path to an audio file.
        sr: Target sample rate (48000 for CLAP).
        max_seconds: Fixed output duration in seconds.

    Returns:
        ``(max_samples,)`` 1-D waveform tensor.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    max_samples = int(max_seconds * sr)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return torch.from_numpy(audio)


def audio_path_to_cache_location(
    audio_path: str | Path,
    cache_dir: str | Path,
    suffix: str = "",
) -> Path:
    cache_dir = Path(cache_dir)
    safe_name = str(Path(audio_path)).replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}{suffix}.pt"


def _waveform_cache_suffix(sr: int) -> str:
    """Return a cache filename suffix that distinguishes waveform sample rates."""
    return f"_wav{sr}"


def _clap_features_cache_suffix(model_name: str) -> str:
    """Return a cache filename suffix scoped to a CLAP model identity.

    Different CLAP checkpoints emit different mel features, so the
    cached dicts are not interchangeable across models.
    """
    safe = model_name.replace("/", "_").replace("\\", "_")
    return f"_clapfeat_{safe}"


def cache_waveforms(
    audio_dir: str | Path,
    cache_dir: str | Path,
    sr: int = 48000,
    max_seconds: float = 10.0,
) -> None:
    """Pre-compute and cache resampled waveforms for all audio files.

    Args:
        audio_dir: Directory containing audio files.
        cache_dir: Directory to write cached ``.pt`` tensors.
        sr: Target sample rate (48000 for CLAP).
        max_seconds: Fixed output duration in seconds.
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    iter_audio_paths = (
        f
        for f in Path(audio_dir).rglob("*")
        if f.is_file() and f.suffix in AUDIO_EXTENSIONS
    )
    suffix = _waveform_cache_suffix(sr)
    logger.info("Caching waveforms (%d Hz) from %s ...", sr, audio_dir)
    for i, audio_path in enumerate(iter_audio_paths, 1):
        dest = audio_path_to_cache_location(
            audio_path=audio_path,
            cache_dir=cache_dir,
            suffix=suffix,
        )
        if dest.exists():
            continue
        waveform = audio_to_waveform(audio_path, sr=sr, max_seconds=max_seconds)
        torch.save(waveform, dest)
        if i % 500 == 0:
            logger.info("[%d] waveforms cached...", i)


def cache_clap_features(
    audio_dir: str | Path,
    cache_dir: str | Path,
    processor: ClapProcessor,
    model_name: str,
    sr: int = 48000,
    max_seconds: float = 10.0,
) -> None:
    """Pre-compute and cache CLAP processor outputs for all audio files.

    Mel-spectrogram extraction is by far the most expensive CPU step
    per training batch. Caching the processor's output dict per audio
    file lets DataLoader workers do nothing more than ``torch.load``,
    keeping the GPU fed.

    Args:
        audio_dir: Directory containing audio files (recursively scanned).
        cache_dir: Directory to write cached ``.pt`` dicts.
        processor: A ClapProcessor matching ``model_name``.
        model_name: HuggingFace ID for the CLAP model; used to scope
            the cache filename so multiple models can coexist.
        sr: Target sample rate (48000 for CLAP).
        max_seconds: Fixed input duration in seconds.
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    iter_audio_paths = (
        f
        for f in Path(audio_dir).rglob("*")
        if f.is_file() and f.suffix in AUDIO_EXTENSIONS
    )
    suffix = _clap_features_cache_suffix(model_name)
    logger.info(
        "Caching CLAP features (%s) from %s ...", model_name, audio_dir
    )
    for i, audio_path in enumerate(iter_audio_paths, 1):
        dest = audio_path_to_cache_location(
            audio_path=audio_path,
            cache_dir=cache_dir,
            suffix=suffix,
        )
        if dest.exists():
            continue
        waveform = audio_to_waveform(audio_path, sr=sr, max_seconds=max_seconds)
        wav_np = waveform.float().numpy()
        raw = processor(
            audio=[wav_np],
            sampling_rate=sr,
            return_tensors="pt",
        )
        cached: dict[str, Tensor] = {
            k: v for k, v in raw.items() if k in CLAP_AUDIO_INPUT_KEYS
        }
        torch.save(cached, dest)
        if i % 500 == 0:
            logger.info("[%d] CLAP feature dicts cached...", i)


_SPLIT_SEED = 42
_EVAL_FRACTION = 0.15


def _apply_split(
    rows: list[dict[str, str]],
    split: str,
) -> list[dict[str, str]]:
    """Filter manifest rows into train or eval partition.

    The split is by unique audio track so that all caption variants for
    a given track stay in the same partition.  Jamendo tracks are always
    assigned to train because their programmatic captions are not useful
    for measuring generalization.
    """
    lp_tracks = sorted(
        {r["audio_path"] for r in rows if r.get("source") != "jamendo"}
    )
    rng = random.Random(_SPLIT_SEED)
    rng.shuffle(lp_tracks)
    n_eval = max(1, int(len(lp_tracks) * _EVAL_FRACTION))
    eval_set = set(lp_tracks[-n_eval:])

    if split == "eval":
        return [r for r in rows if r["audio_path"] in eval_set]
    if split == "train":
        return [r for r in rows if r["audio_path"] not in eval_set]
    msg = f"split must be 'train', 'eval', or None, got {split!r}"
    raise ValueError(msg)


_WAVEFORM_SR = 48000


class AudioTextDataset(Dataset[tuple[Tensor, str]]):
    """Dataset of (audio_waveform, text) pairs at 48 kHz for CLAP.

    Reads a CSV manifest with ``audio_path`` and ``text`` columns.
    Audio is loaded from cached ``.pt`` waveform tensors.

    Args:
        manifest_path: Path to CSV file with ``audio_path`` and
            ``text`` columns.
        cache_dir: Directory containing cached ``.pt`` tensors.
        split: ``"train"``, ``"eval"``, or ``None``.  When set,
            partitions by track so all caption variants stay together.
            Jamendo tracks are always train-only.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        split: str | None = None,
    ) -> None:
        with Path(manifest_path).open(mode="r", encoding="utf8") as manifest_f:
            reader = csv.DictReader(manifest_f)
            rows = list(reader)
        if split is not None:
            rows = _apply_split(rows, split)
        cache_suffix = _waveform_cache_suffix(_WAVEFORM_SR)
        self.pairs = [
            (
                audio_path_to_cache_location(
                    audio_path=row["audio_path"],
                    cache_dir=cache_dir,
                    suffix=cache_suffix,
                ),
                row["text"],
            )
            for row in rows
        ]
        # Track IDs for UniqueAudioBatchSampler. Uses audio_path as
        # the unique identifier for each underlying audio file.
        self.track_ids: list[str] = [row["audio_path"] for row in rows]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> tuple[Tensor, str]:  # type: ignore[override]
        """Return (waveform, text) for the given index."""
        cache_location, text = self.pairs[idx]
        audio = torch.load(cache_location, weights_only=True)
        return audio, text


ClapBatch = tuple[dict[str, Tensor], dict[str, Tensor]]
ClapFeaturesItem = tuple[dict[str, Tensor], str]


class ClapFeaturesTextDataset(Dataset[ClapFeaturesItem]):
    """Dataset of (cached CLAP audio inputs, text) pairs.

    Reads a CSV manifest with ``audio_path`` and ``text`` columns and
    loads the matching cached CLAP feature dict produced by
    :func:`cache_clap_features`. Eliminates per-step mel-spectrogram
    extraction so DataLoader workers do nothing but ``torch.load``.

    Args:
        manifest_path: Path to CSV file with ``audio_path`` and
            ``text`` columns.
        cache_dir: Directory containing cached ``.pt`` feature dicts.
        model_name: HuggingFace CLAP model ID; must match the one
            passed to ``cache_clap_features``.
        split: ``"train"``, ``"eval"``, or ``None``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        model_name: str,
        split: str | None = None,
    ) -> None:
        with Path(manifest_path).open(mode="r", encoding="utf8") as manifest_f:
            reader = csv.DictReader(manifest_f)
            rows = list(reader)
        if split is not None:
            rows = _apply_split(rows, split)
        suffix = _clap_features_cache_suffix(model_name)
        self.pairs: list[tuple[Path, str]] = [
            (
                audio_path_to_cache_location(
                    audio_path=row["audio_path"],
                    cache_dir=cache_dir,
                    suffix=suffix,
                ),
                row["text"],
            )
            for row in rows
        ]
        self.track_ids: list[str] = [row["audio_path"] for row in rows]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> ClapFeaturesItem:  # type: ignore[override]
        cache_location, text = self.pairs[idx]
        audio_inputs: dict[str, Tensor] = torch.load(
            cache_location, weights_only=True
        )
        return audio_inputs, text


def make_clap_collate_fn(
    processor: ClapProcessor,
) -> Callable[[list[ClapFeaturesItem]], ClapBatch]:
    """Build a collate_fn for cached CLAP feature dicts + raw text.

    Audio-side work is already done by :func:`cache_clap_features`, so
    the collate_fn only stacks the cached dicts and tokenizes the text
    batch (cheap; runs in worker processes).

    Args:
        processor: A ClapProcessor used for text tokenization.

    Returns:
        Callable suitable for ``DataLoader(collate_fn=...)`` that
        returns ``(audio_inputs, text_inputs)`` dicts of tensors ready
        to feed to ``ClapModel.get_*_features``.
    """

    def collate(batch: list[ClapFeaturesItem]) -> ClapBatch:
        audio_dicts = [d for d, _ in batch]
        texts = [t for _, t in batch]
        keys = audio_dicts[0].keys()
        audio_inputs: dict[str, Tensor] = {
            k: torch.cat([d[k] for d in audio_dicts], dim=0) for k in keys
        }
        text_raw = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {
            k: v for k, v in text_raw.items() if k in CLAP_TEXT_INPUT_KEYS
        }
        return audio_inputs, text_inputs

    return collate


class UniqueAudioBatchSampler(Sampler[list[int]]):
    """Batch sampler that prevents the same audio track from appearing
    twice in a batch.

    Rows sharing an audio_path have identical audio embeddings, so
    allowing them in the same batch creates false negatives in the
    InfoNCE loss. This sampler uses round-robin interleaving across
    caption groups so that within any window of N_unique_tracks
    consecutive indices, each track appears exactly once. Because
    batch_size is much smaller than N_unique_tracks, batches will
    never contain duplicate audio tracks.

    Args:
        track_ids: Per-row audio track identifier (audio_path).
            Rows with the same track_id share the same audio file.
        batch_size: Number of samples per batch.
        drop_last: If True, drop the final batch when smaller than
            batch_size.
    """

    def __init__(
        self,
        track_ids: list[str],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        groups: dict[str, list[int]] = {}
        for idx, tid in enumerate(track_ids):
            groups.setdefault(tid, []).append(idx)
        self._groups = list(groups.values())
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        # One item per unique track per epoch.
        n = len(self._groups)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        # Pick one caption per track at random each epoch so that over
        # many epochs all caption variants are seen, while within any
        # single batch each audio track appears at most once.
        indices = [random.choice(g) for g in self._groups]
        random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch
