"""Audio preprocessing and dataset for text-audio pairs."""

from __future__ import annotations

import csv
import logging
import math
import random
from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# AST positional embeddings are fixed at 1024 time frames × 128 mel bins.
# These normalization constants come from ASTFeatureExtractor's defaults.
_AST_MAX_FRAMES = 1024
_AST_MEAN = -4.2677393
_AST_STD = 4.5689974

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}


def audio_to_mel_spectrogram(
    path: str | Path,
    n_mels: int = 128,
    sr: int = 16000,
    max_seconds: float = 10.0,
) -> Tensor:
    """Convert an audio file to a mel spectrogram tensor.

    Uses librosa to load and convert the audio. The output is a
    2D tensor of shape ``(n_mels, time_frames)`` where time_frames
    depends on the audio duration (capped at ``max_seconds``).

    Args:
        path: Path to an audio file (wav, mp3, flac, etc.).
        n_mels: Number of mel frequency bins.
        sr: Target sample rate (audio is resampled if needed).
        max_seconds: Maximum audio duration in seconds.

    Returns:
        ``(n_mels, time_frames)`` mel spectrogram tensor.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    max_samples = int(max_seconds * sr)
    audio = audio[:max_samples]
    # 25ms window / 10ms hop matches ASTFeatureExtractor defaults.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160,
    )
    log_mel = librosa.power_to_db(mel).T  # (time, n_mels)
    # Pad or truncate to AST's fixed positional embedding length.
    t = log_mel.shape[0]
    if t < _AST_MAX_FRAMES:
        pad = np.zeros((_AST_MAX_FRAMES - t, n_mels), dtype=log_mel.dtype)
        log_mel = np.concatenate([log_mel, pad], axis=0)
    else:
        log_mel = log_mel[:_AST_MAX_FRAMES]
    # Normalize with AST's dataset-level statistics.
    log_mel = (log_mel - _AST_MEAN) / (_AST_STD * 2)
    return torch.from_numpy(log_mel)


def audio_to_waveform(
    path: str | Path,
    sr: int = 24000,
    max_seconds: float = 10.0,
) -> Tensor:
    """Load an audio file as a raw waveform tensor.

    Resamples to ``sr``, truncates or zero-pads to exactly
    ``max_seconds`` so all waveforms in a batch have equal length.

    Args:
        path: Path to an audio file.
        sr: Target sample rate (24000 for MERT).
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


def cache_spectrograms(
    audio_dir: str | Path,
    cache_dir: str | Path,
    n_mels: int = 128,
    sr: int = 16000,
    max_seconds: float = 10.0,
) -> None:
    """Pre-compute and cache mel spectrograms for all audio files.

    Walks ``audio_dir`` for audio files, converts each to a mel
    spectrogram, and saves the tensor to ``cache_dir`` as a ``.pt``
    file with the same stem.

    Args:
        audio_dir: Directory containing audio files.
        cache_dir: Directory to write cached ``.pt`` tensors.
        n_mels: Number of mel frequency bins.
        sr: Target sample rate.
        max_seconds: Maximum audio duration in seconds.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    iter_audio_paths = (
        f
        for f in Path(audio_dir).rglob("*")
        if f.is_file() and f.suffix in AUDIO_EXTENSIONS
    )
    logger.info("Caching spectrograms from %s ...", audio_dir)
    for i, audio_path in enumerate(iter_audio_paths, 1):
        spectrogram = audio_to_mel_spectrogram(
            audio_path,
            n_mels=n_mels,
            sr=sr,
            max_seconds=max_seconds,
        )
        torch.save(
            spectrogram,
            audio_path_to_cache_location(
                audio_path=audio_path,
                cache_dir=cache_dir,
            ),
        )
        if i % 500 == 0:
            logger.info("[%d] spectrograms cached...", i)


def cache_waveforms(
    audio_dir: str | Path,
    cache_dir: str | Path,
    sr: int = 24000,
    max_seconds: float = 10.0,
) -> None:
    """Pre-compute and cache resampled waveforms for all audio files.

    Args:
        audio_dir: Directory containing audio files.
        cache_dir: Directory to write cached ``.pt`` tensors.
        sr: Target sample rate (24000 for MERT).
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


def _waveform_cache_suffix(sr: int) -> str:
    """Return a cache filename suffix that distinguishes waveform sample rates."""
    return f"_wav{sr}"


def apply_spec_augment(
    spec: Tensor,
    freq_masks: int,
    freq_width: int,
    time_masks: int,
    time_width: int,
) -> Tensor:
    """Apply SpecAugment time and frequency masking to a spectrogram.

    Randomly zeros out frequency strips (horizontal) and time strips
    (vertical) in the spectrogram. Operates on a clone so the cached
    tensor on disk is never modified.

    Args:
        spec: Spectrogram tensor of shape ``(time_frames, n_mels)``.
        freq_masks: Number of frequency mask strips to apply.
        freq_width: Maximum width of each frequency mask (mel bins).
        time_masks: Number of time mask strips to apply.
        time_width: Maximum width of each time mask (frames).

    Returns:
        Augmented spectrogram tensor with the same shape.
    """
    spec = spec.clone()
    t_frames, n_mels = spec.shape
    for _ in range(freq_masks):
        f = random.randint(0, freq_width)
        f0 = random.randint(0, max(n_mels - f, 0))
        spec[:, f0 : f0 + f] = 0.0
    for _ in range(time_masks):
        t = random.randint(0, time_width)
        t0 = random.randint(0, max(t_frames - t, 0))
        spec[t0 : t0 + t, :] = 0.0
    return spec


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


class AudioTextDataset(Dataset[tuple[Tensor, str]]):
    """Dataset of (audio_tensor, text) pairs.

    Reads a CSV manifest with at least ``audio_path`` and ``text``
    columns. Audio tensors are loaded from the cache directory as
    ``.pt`` files, either as mel spectrograms (for AST) or raw
    waveforms (for MERT).

    Args:
        manifest_path: Path to CSV file with ``audio_path`` and
            ``text`` columns.
        cache_dir: Directory containing cached ``.pt`` tensors.
        split: ``"train"``, ``"eval"``, or ``None``.  When set,
            partitions by track so all caption variants stay together.
            Jamendo tracks are always train-only.
        augment: If True, apply SpecAugment masking on each load
            (spectrograms only; ignored for waveforms).
        waveform_sr: If set, load cached waveforms at this sample
            rate instead of spectrograms.
        spec_aug_freq_masks: Number of frequency mask strips.
        spec_aug_freq_width: Max width of each frequency mask.
        spec_aug_time_masks: Number of time mask strips.
        spec_aug_time_width: Max width of each time mask.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        split: str | None = None,
        augment: bool = False,
        waveform_sr: int | None = None,
        spec_aug_freq_masks: int = 2,
        spec_aug_freq_width: int = 27,
        spec_aug_time_masks: int = 2,
        spec_aug_time_width: int = 100,
    ) -> None:
        with Path(manifest_path).open(mode="r", encoding="utf8") as manifest_f:
            reader = csv.DictReader(manifest_f)
            rows = list(reader)
        if split is not None:
            rows = _apply_split(rows, split)
        cache_suffix = (
            _waveform_cache_suffix(waveform_sr) if waveform_sr else ""
        )
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
        self.use_waveforms = waveform_sr is not None
        self.augment = augment
        self.spec_aug_freq_masks = spec_aug_freq_masks
        self.spec_aug_freq_width = spec_aug_freq_width
        self.spec_aug_time_masks = spec_aug_time_masks
        self.spec_aug_time_width = spec_aug_time_width

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> tuple[Tensor, str]:  # type: ignore[override]
        """Return (audio_tensor, text) for the given index."""
        cache_location, text = self.pairs[idx]
        audio = torch.load(cache_location, weights_only=True)
        if not self.use_waveforms and self.augment:
            audio = apply_spec_augment(
                audio,
                freq_masks=self.spec_aug_freq_masks,
                freq_width=self.spec_aug_freq_width,
                time_masks=self.spec_aug_time_masks,
                time_width=self.spec_aug_time_width,
            )
        return audio, text


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
