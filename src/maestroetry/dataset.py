"""Audio preprocessing and dataset for text-audio pairs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

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


def audio_path_to_cache_location(
    audio_path: str | Path, cache_dir: str | Path
) -> Path:
    cache_dir = Path(cache_dir)
    safe_name = str(Path(audio_path)).replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}.pt"


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


class AudioTextDataset(Dataset[tuple[Tensor, str]]):
    """Dataset of (spectrogram, text) pairs.

    Reads a CSV manifest with at least ``audio_path`` and ``text``
    columns. Spectrograms are loaded from the cache directory as
    ``.pt`` files.

    Args:
        manifest_path: Path to CSV file with ``audio_path`` and
            ``text`` columns.
        cache_dir: Directory containing cached ``.pt`` spectrograms.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
    ) -> None:
        with Path(manifest_path).open(mode="r", encoding="utf8") as manifest_f:
            reader = csv.DictReader(manifest_f)
            self.pairs = [
                (
                    audio_path_to_cache_location(
                        audio_path=row["audio_path"], cache_dir=cache_dir
                    ),
                    row["text"],
                )
                for row in reader
            ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> tuple[Tensor, str]:  # type: ignore[override]
        """Return (spectrogram, text) for the given index."""
        cache_location, text = self.pairs[idx]
        spectrogram = torch.load(cache_location, weights_only=True)
        return spectrogram, text
