"""Tests for the SDD + FMA ingest pipeline."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from unittest.mock import patch

from maestroetry.ingest import (
    _fma_track_id_to_path,
    build_caption,
    ingest,
)


class TestBuildCaption:
    """Tests for caption generation from FMA metadata."""

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
            genres=[], title="Untitled", tags=["piano"],
        )
        assert "unknown genre" in caption

    def test_empty_tags_uses_fallback(self) -> None:
        random.seed(0)
        caption = build_caption(
            genres=["Rock"], title="Song", tags=[],
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


class TestFmaTrackIdToPath:
    """Tests for FMA track ID path resolution."""

    def test_existing_track(self, tmp_path: Path) -> None:
        # Create the expected FMA directory structure
        subdir = tmp_path / "002"
        subdir.mkdir()
        mp3 = subdir / "002345.mp3"
        mp3.write_bytes(b"fake mp3")

        result = _fma_track_id_to_path(2345, tmp_path)
        assert result == mp3

    def test_missing_track_returns_none(
        self, tmp_path: Path,
    ) -> None:
        result = _fma_track_id_to_path(999999, tmp_path)
        assert result is None


class TestManifestWriting:
    """Tests for the ingest orchestrator's manifest output."""

    def test_manifest_csv_format(self, tmp_path: Path) -> None:
        """Verify manifest has correct columns when both
        ingest functions return rows."""
        mock_sdd_rows = [
            {
                "audio_path": "data/audio/sdd/000001.wav",
                "text": "A mellow jazz tune",
                "source": "sdd",
            },
        ]
        mock_fma_rows = [
            {
                "audio_path": "data/audio/fma/000100.wav",
                "text": "A Rock track with varied instrumentation",
                "source": "fma",
            },
        ]

        with (
            patch(
                "maestroetry.ingest.ingest_sdd",
                return_value=mock_sdd_rows,
            ),
            patch(
                "maestroetry.ingest.ingest_fma",
                return_value=mock_fma_rows,
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
        assert rows[0]["source"] == "sdd"
        assert rows[1]["source"] == "fma"

    def test_sdd_only_flag(self, tmp_path: Path) -> None:
        with (
            patch(
                "maestroetry.ingest.ingest_sdd",
                return_value=[],
            ) as mock_sdd,
            patch(
                "maestroetry.ingest.ingest_fma",
                return_value=[],
            ) as mock_fma,
        ):
            ingest(data_dir=tmp_path, sdd_only=True)

        mock_sdd.assert_called_once()
        mock_fma.assert_not_called()

    def test_fma_only_flag(self, tmp_path: Path) -> None:
        with (
            patch(
                "maestroetry.ingest.ingest_sdd",
                return_value=[],
            ) as mock_sdd,
            patch(
                "maestroetry.ingest.ingest_fma",
                return_value=[],
            ) as mock_fma,
        ):
            ingest(data_dir=tmp_path, fma_only=True)

        mock_sdd.assert_not_called()
        mock_fma.assert_called_once()
