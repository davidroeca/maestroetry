"""Download LP-MusicCaps and MTG-Jamendo datasets for training."""

from __future__ import annotations

import csv
import io
import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Caption templates for programmatic Jamendo captions.
# Each template is a format string accepting genre and tags.
_CAPTION_TEMPLATES: list[str] = [
    "A {genre} track with {tags}",
    "An energetic {genre} piece featuring {tags}",
    "A {genre} song characterized by {tags}",
    "{genre} music with {tags}",
    "A {genre} composition with elements of {tags}",
    "A track in the {genre} style, featuring {tags}",
    "{genre} recording showcasing {tags}",
    "A {genre} number driven by {tags}",
]


def build_caption(
    genres: list[str],
    title: str,
    tags: list[str],
) -> str:
    """Build a natural-language caption from metadata.

    Produces descriptive captions from genre and tag labels
    so the model sees a uniform distribution of caption styles.

    Args:
        genres: Genre labels for the track.
        title: Track title (currently unused but available
            for future template enrichment).
        tags: Instrumentation or mood tags.

    Returns:
        A single descriptive sentence.
    """
    genre_str = genres[0] if genres else "unknown genre"
    if tags:
        tags_str = ", ".join(tags[:3])
    else:
        tags_str = "varied instrumentation"
    template = random.choice(_CAPTION_TEMPLATES)
    return template.format(genre=genre_str, tags=tags_str)


def _decode_audio_struct(audio_struct: dict) -> tuple[np.ndarray, int]:
    """Decode a PyArrow audio struct to a numpy array and sample rate.

    HuggingFace datasets store audio either as a decoded float array
    (``{"array": [...], "sampling_rate": int}``) or as encoded bytes
    (``{"bytes": binary}``). We read the raw PyArrow struct directly
    to avoid the datasets Audio feature decoder, which pulls in torchcodec.

    Args:
        audio_struct: Dict from ``pa_table["audio"][i].as_py()``.

    Returns:
        Tuple of (audio_array float32, sampling_rate).
    """
    if "array" in audio_struct:
        return np.array(audio_struct["array"], dtype=np.float32), audio_struct["sampling_rate"]
    audio_array, sr = sf.read(io.BytesIO(audio_struct["bytes"]))
    return np.array(audio_array, dtype=np.float32), int(sr)


def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert audio to 16kHz mono wav via ffmpeg.

    Args:
        src: Source audio file path.
        dst: Destination wav file path.

    Returns:
        True if conversion succeeded.
    """
    if dst.exists():
        return True
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=30,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("Failed to convert %s", src)
        return False
    return dst.exists()


def _save_audio_array_as_wav(
    audio_array: np.ndarray,
    sampling_rate: int,
    dest: Path,
) -> None:
    """Write a numpy audio array to WAV at 16kHz mono.

    Resamples if the source rate differs from 16kHz.

    Args:
        audio_array: 1-D float audio samples.
        sampling_rate: Sample rate of the input array.
        dest: Destination WAV file path.
    """
    target_rate = 16000
    if sampling_rate != target_rate:
        from scipy.signal import resample

        num_samples = int(len(audio_array) * target_rate / sampling_rate)
        audio_array = resample(audio_array, num_samples)
        sampling_rate = target_rate

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio_array, sampling_rate, subtype="PCM_16")


def ingest_lp_musiccaps(
    data_dir: str | Path = "data",
    max_samples: int | None = None,
    split: str = "train",
) -> list[dict[str, str]]:
    """Download LP-MusicCaps-MTT and extract audio + captions.

    Each row has an ``audio`` struct and ``texts`` (list of 4 caption
    variants); one caption is picked at random for diversity.

    Args:
        data_dir: Root data directory.
        max_samples: Cap on number of samples to process.
        split: Dataset split to load.

    Returns:
        List of manifest row dicts with audio_path, text, source.
    """
    from datasets import load_dataset

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio" / "lp_musiccaps"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading LP-MusicCaps-MTT dataset (split=%s)...", split)
    ds = load_dataset(
        "mulab-mir/lp-music-caps-magnatagatune-3k",
        split=split,
    )
    pa_table = ds.data.table
    n = len(pa_table) if max_samples is None else min(len(pa_table), max_samples)

    rows: list[dict[str, str]] = []
    for i in range(n):
        wav_path = audio_dir / f"{i:06d}.wav"
        if not wav_path.exists():
            audio_array, sr = _decode_audio_struct(pa_table["audio"][i].as_py())
            _save_audio_array_as_wav(audio_array, sr, wav_path)

        caption = random.choice(pa_table["texts"][i].as_py())
        rows.append(
            {
                "audio_path": str(wav_path),
                "text": caption,
                "source": "lp_musiccaps",
            }
        )
        if len(rows) % 100 == 0:
            logger.info("[%d] LP-MusicCaps tracks processed...", len(rows))

    logger.info("LP-MusicCaps ingest complete: %d samples", len(rows))
    return rows


def ingest_jamendo(
    data_dir: str | Path = "data",
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    """Download MTG-Jamendo and generate captions from tags.

    Each row has an ``audio`` struct, ``genre``, ``instrument``, and
    ``mood_theme`` lists; captions are generated via ``build_caption``.

    Args:
        data_dir: Root data directory.
        max_samples: Cap on number of samples to process.

    Returns:
        List of manifest row dicts with audio_path, text, source.
    """
    from datasets import load_dataset

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio" / "jamendo"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MTG-Jamendo dataset...")
    ds = load_dataset(
        "vtsouval/mtg_jamendo_autotagging",
        split="train",
    )
    pa_table = ds.data.table
    n = len(pa_table) if max_samples is None else min(len(pa_table), max_samples)

    rows: list[dict[str, str]] = []
    for i in range(n):
        wav_path = audio_dir / f"{i:06d}.wav"
        if not wav_path.exists():
            audio_array, sr = _decode_audio_struct(pa_table["audio"][i].as_py())
            _save_audio_array_as_wav(audio_array, sr, wav_path)

        caption = build_caption(
            genres=pa_table["genre"][i].as_py(),
            title="",
            tags=pa_table["instrument"][i].as_py()
            + pa_table["mood_theme"][i].as_py(),
        )
        rows.append(
            {
                "audio_path": str(wav_path),
                "text": caption,
                "source": "jamendo",
            }
        )
        if len(rows) % 100 == 0:
            logger.info("[%d] Jamendo tracks processed...", len(rows))

    logger.info("Jamendo ingest complete: %d samples", len(rows))
    return rows


def ingest(
    data_dir: str | Path = "data",
    max_samples_lp: int | None = None,
    max_samples_jamendo: int | None = None,
    lp_only: bool = False,
    jamendo_only: bool = False,
) -> None:
    """Orchestrate LP-MusicCaps and Jamendo ingest.

    Calls ingest_lp_musiccaps and/or ingest_jamendo and writes
    a combined manifest.csv with audio_path, text, and source
    columns.

    Args:
        data_dir: Root data directory.
        max_samples_lp: Cap on LP-MusicCaps samples.
        max_samples_jamendo: Cap on Jamendo samples.
        lp_only: If True, only ingest LP-MusicCaps.
        jamendo_only: If True, only ingest Jamendo.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"
    all_rows: list[dict[str, str]] = []

    if not jamendo_only:
        all_rows.extend(
            ingest_lp_musiccaps(data_dir, max_samples=max_samples_lp)
        )

    if not lp_only:
        all_rows.extend(
            ingest_jamendo(data_dir, max_samples=max_samples_jamendo)
        )

    fieldnames = ["audio_path", "text", "source"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "Done. %d total samples. Manifest: %s",
        len(all_rows),
        manifest_path,
    )
