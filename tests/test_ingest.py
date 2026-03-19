"""Tests for the LP-MusicCaps + Jamendo ingest pipeline."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
from maestroetry.ingest import (
    _save_audio_array_as_wav,
    build_caption,
    ingest,
)


class TestBuildCaption:
    """Tests for caption generation from metadata."""

    def test_single_genre_with_tags(self) -> None:
        random.seed(42)
        caption = build_caption(
            genres=["Hip-Hop"],
            title="Test Track",
            tags=["electronic", "bass"],
        )
        assert "Hip-Hop" in caption
        assert "electronic" in caption

    def test_empty_genres_uses_fallback(self) -> None:
        random.seed(0)
        caption = build_caption(
            genres=[],
            title="Untitled",
            tags=["piano"],
        )
        assert "unknown genre" in caption

    def test_empty_tags_uses_fallback(self) -> None:
        random.seed(0)
        caption = build_caption(
            genres=["Rock"],
            title="Song",
            tags=[],
        )
        assert "varied instrumentation" in caption

    def test_tags_limited_to_three(self) -> None:
        random.seed(0)
        caption = build_caption(
            genres=["Jazz"],
            title="Long Tags",
            tags=["sax", "piano", "drums", "bass", "trumpet"],
        )
        # Should only include at most 3 tags
        assert "trumpet" not in caption
        assert "bass" not in caption

    def test_returns_string(self) -> None:
        result = build_caption(["Pop"], "Title", ["guitar"])
        assert isinstance(result, str)
        assert len(result) > 0


class TestSaveAudioArrayAsWav:
    """Tests for writing numpy arrays to WAV files."""

    def test_writes_valid_wav(self, tmp_path: Path) -> None:
        import soundfile as sf

        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(
            np.float32
        )
        dest = tmp_path / "test.wav"

        _save_audio_array_as_wav(audio, 16000, dest)

        assert dest.exists()
        data, sr = sf.read(str(dest))
        assert sr == 16000
        assert len(data) == 16000

    def test_resamples_non_16k(self, tmp_path: Path) -> None:
        import soundfile as sf

        # 1 second at 22050 Hz
        audio = np.zeros(22050, dtype=np.float32)
        dest = tmp_path / "resampled.wav"

        _save_audio_array_as_wav(audio, 22050, dest)

        assert dest.exists()
        data, sr = sf.read(str(dest))
        assert sr == 16000


class TestManifestWriting:
    """Tests for the ingest orchestrator's manifest output."""

    def test_manifest_csv_format(self, tmp_path: Path) -> None:
        """Verify manifest has correct columns when both
        ingest functions return rows."""
        mock_lp_rows = [
            {
                "audio_path": "data/audio/lp_musiccaps/000001.wav",
                "text": "A mellow jazz tune with piano",
                "source": "lp_musiccaps",
            },
        ]
        mock_jamendo_rows = [
            {
                "audio_path": "data/audio/jamendo/000100.wav",
                "text": "A Rock track with varied instrumentation",
                "source": "jamendo",
            },
        ]

        with (
            patch(
                "maestroetry.ingest.ingest_lp_musiccaps",
                return_value=mock_lp_rows,
            ),
            patch(
                "maestroetry.ingest.ingest_jamendo",
                return_value=mock_jamendo_rows,
            ),
        ):
            ingest(data_dir=tmp_path)

        manifest = tmp_path / "manifest.csv"
        assert manifest.exists()

        with manifest.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert set(reader.fieldnames) == {
            "audio_path",
            "text",
            "source",
        }
        assert rows[0]["source"] == "lp_musiccaps"
        assert rows[1]["source"] == "jamendo"

    def test_lp_only_flag(self, tmp_path: Path) -> None:
        with (
            patch(
                "maestroetry.ingest.ingest_lp_musiccaps",
                return_value=[],
            ) as mock_lp,
            patch(
                "maestroetry.ingest.ingest_jamendo",
                return_value=[],
            ) as mock_jamendo,
        ):
            ingest(data_dir=tmp_path, lp_only=True)

        mock_lp.assert_called_once()
        mock_jamendo.assert_not_called()

    def test_jamendo_only_flag(self, tmp_path: Path) -> None:
        with (
            patch(
                "maestroetry.ingest.ingest_lp_musiccaps",
                return_value=[],
            ) as mock_lp,
            patch(
                "maestroetry.ingest.ingest_jamendo",
                return_value=[],
            ) as mock_jamendo,
        ):
            ingest(data_dir=tmp_path, jamendo_only=True)

        mock_lp.assert_not_called()
        mock_jamendo.assert_called_once()
