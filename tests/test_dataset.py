"""Tests for dataset split logic."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from maestroetry.dataset import _apply_split


def _make_manifest(tmp: Path, rows: list[dict[str, str]]) -> Path:
    """Write a minimal manifest CSV and return its path."""
    manifest = tmp / "manifest.csv"
    with manifest.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _build_rows() -> list[dict[str, str]]:
    """Build a small manifest with mixed sources and multi-caption tracks."""
    rows: list[dict[str, str]] = []
    # 20 LP-MusicCaps tracks, each with 4 captions
    for i in range(20):
        for cap in range(4):
            rows.append({
                "audio_path": f"lp_track_{i:03d}.wav",
                "text": f"Caption {cap} for LP track {i}",
                "source": "lp_musiccaps",
            })
    # 10 Jamendo tracks, 1 caption each
    for i in range(10):
        rows.append({
            "audio_path": f"jamendo_track_{i:03d}.wav",
            "text": f"Jamendo caption for track {i}",
            "source": "jamendo",
        })
    return rows


class TestApplySplit:
    """Tests for the _apply_split function."""

    def test_no_track_overlap_between_splits(self) -> None:
        rows = _build_rows()
        train_rows = _apply_split(rows, "train")
        eval_rows = _apply_split(rows, "eval")
        train_tracks = {r["audio_path"] for r in train_rows}
        eval_tracks = {r["audio_path"] for r in eval_rows}
        assert train_tracks.isdisjoint(eval_tracks)

    def test_jamendo_always_in_train(self) -> None:
        rows = _build_rows()
        eval_rows = _apply_split(rows, "eval")
        eval_sources = {r["source"] for r in eval_rows}
        assert "jamendo" not in eval_sources

        train_rows = _apply_split(rows, "train")
        jamendo_in_train = {
            r["audio_path"]
            for r in train_rows
            if r["source"] == "jamendo"
        }
        jamendo_total = {
            r["audio_path"]
            for r in rows
            if r["source"] == "jamendo"
        }
        assert jamendo_in_train == jamendo_total

    def test_all_lp_rows_covered(self) -> None:
        rows = _build_rows()
        train_rows = _apply_split(rows, "train")
        eval_rows = _apply_split(rows, "eval")
        lp_total = {
            r["audio_path"]
            for r in rows
            if r["source"] == "lp_musiccaps"
        }
        lp_combined = {
            r["audio_path"] for r in train_rows if r["source"] == "lp_musiccaps"
        } | {r["audio_path"] for r in eval_rows}
        assert lp_combined == lp_total

    def test_all_caption_variants_stay_together(self) -> None:
        rows = _build_rows()
        eval_rows = _apply_split(rows, "eval")
        eval_tracks = {r["audio_path"] for r in eval_rows}
        for track in eval_tracks:
            captions = [r["text"] for r in eval_rows if r["audio_path"] == track]
            expected = [r["text"] for r in rows if r["audio_path"] == track]
            assert captions == expected

    def test_deterministic(self) -> None:
        rows = _build_rows()
        eval_a = _apply_split(rows, "eval")
        eval_b = _apply_split(rows, "eval")
        assert eval_a == eval_b

    def test_eval_fraction_approximately_15_percent(self) -> None:
        rows = _build_rows()
        eval_rows = _apply_split(rows, "eval")
        eval_tracks = {r["audio_path"] for r in eval_rows}
        # 20 LP tracks, 15% = 3
        assert len(eval_tracks) == 3

    def test_invalid_split_raises(self) -> None:
        rows = _build_rows()
        with pytest.raises(ValueError, match="split must be"):
            _apply_split(rows, "test")
