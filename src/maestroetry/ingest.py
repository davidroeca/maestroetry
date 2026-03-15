"""Download and structure MusicCaps for training."""

from __future__ import annotations

import csv
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _download_clip(
    ytid: str,
    start_s: float,
    end_s: float,
    out_path: Path,
) -> bool:
    """Download a YouTube clip and extract the target segment.

    Args:
        ytid: YouTube video ID.
        start_s: Segment start time in seconds.
        end_s: Segment end time in seconds.
        out_path: Output path for the audio file.

    Returns:
        True if the download succeeded, False otherwise.
    """
    if out_path.exists():
        logger.debug("Already exists: %s", out_path)
        return True
    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end_s - start_s
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "full.%(ext)s"
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--output",
            str(tmp_path),
            "--quiet",
            "--no-warnings",
            url,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("Failed to download %s", ytid)
            return False

        downloaded = list(Path(tmp).glob("full.*"))
        if not downloaded:
            logger.warning("No output file for %s", ytid)
            return False

        # Extract the target segment with ffmpeg
        cmd_trim = [
            "ffmpeg",
            "-y",
            "-i",
            str(downloaded[0]),
            "-ss",
            str(start_s),
            "-t",
            str(duration),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd_trim,
                check=True,
                timeout=30,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("Failed to trim %s", ytid)
            return False

    return out_path.exists()


def ingest_musiccaps(
    data_dir: str | Path = "data",
    max_samples: int | None = None,
) -> None:
    """Download MusicCaps and produce a manifest CSV.

    Downloads the MusicCaps dataset from HuggingFace, fetches
    audio clips from YouTube via yt-dlp, trims them to the
    annotated segment, and writes a manifest CSV compatible
    with AudioTextDataset.

    Args:
        data_dir: Root data directory. Audio goes into
            ``data_dir/audio/`` and the manifest is written
            to ``data_dir/manifest.csv``.
        max_samples: If set, limit to this many samples
            (useful for testing the pipeline).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install ingest dependencies: uv pip install maestroetry[ingest]"
        ) from exc

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.csv"

    logger.info("Loading MusicCaps metadata from HuggingFace...")
    ds = load_dataset("google/MusicCaps", split="train")

    rows: list[dict[str, str]] = []
    total = len(ds)
    if max_samples is not None:
        total = min(total, max_samples)

    for i, example in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        ytid = example["ytid"]
        start_s = example["start_s"]
        end_s = example["end_s"]
        caption = example["caption"]

        out_path = audio_dir / f"{ytid}.wav"
        logger.info("[%d/%d] Downloading %s...", i + 1, total, ytid)

        if _download_clip(ytid, start_s, end_s, out_path):
            rows.append(
                {
                    "audio_path": str(out_path),
                    "text": caption,
                }
            )
        else:
            logger.warning("[%d/%d] Skipped %s", i + 1, total, ytid)

    with manifest_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Done. %d/%d clips downloaded. Manifest: %s",
        len(rows),
        total,
        manifest_path,
    )
