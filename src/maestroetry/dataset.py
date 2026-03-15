"""Audio preprocessing and dataset for text-audio pairs."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


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
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """Return (spectrogram, text) for the given index."""
        raise NotImplementedError
