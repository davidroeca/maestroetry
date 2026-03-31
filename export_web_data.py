"""Export text projection weights and audio embeddings for the web demo.

Reuses build_model and _strip_compile_prefix from eval_checkpoints.py.
Produces three files in web/static/data/:
  - text_projection.json  (weights for the text projection head)
  - audio_embeddings.json (15 x 256-d float arrays)
  - tracks.json           (metadata manifest)

Usage:
    uv run export_web_data.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/default.toml \\
        --audio-dir demo_audio/ \\
        --tracks-csv demo_tracks.csv \\
        --output-dir web/static/data/

See manual_download.md for instructions on obtaining the demo audio files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval_checkpoints import _strip_compile_prefix, build_model
from maestroetry.config import load_config
from maestroetry.dataset import audio_to_mel_spectrogram
from maestroetry.encoders import encode_audio, encode_text
from maestroetry.projection import ContrastiveModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_projection_layers(head: nn.Sequential) -> list[dict[str, list]]:
    """Return a list of {weight, bias} dicts for each Linear layer in order."""
    layers = []
    for module in head.net:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().cpu().float().tolist(),
                    "bias": module.bias.detach().cpu().float().tolist(),
                }
            )
    return layers


def numpy_project(x: np.ndarray, layers: list[dict]) -> np.ndarray:
    """NumPy reimplementation of ProjectionHead: linear -> ReLU -> ... -> L2 norm.

    Dropout is skipped (inference only). ReLU is applied after every linear
    layer except the last, then output is L2-normalized.
    """
    h = x.astype(np.float32)
    for i, layer in enumerate(layers):
        w = np.array(layer["weight"], dtype=np.float32)  # (out, in)
        b = np.array(layer["bias"], dtype=np.float32)
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    return h / np.maximum(norm, 1e-12)


def validate_text_projection(model: ContrastiveModel, layers: list[dict]) -> None:
    """Assert that numpy_project matches torch ProjectionHead on test inputs."""
    test_texts = [
        "peaceful piano music",
        "energetic orchestral strings",
        "melancholy slow melody",
        "aggressive martial drums",
        "bright joyful violin",
    ]
    raw = encode_text(model.text_encoder, test_texts)
    raw_np = raw.cpu().float().numpy()

    with torch.inference_mode():
        py_embeds = model.text_projection_head(raw).cpu().float().numpy()

    np_embeds = numpy_project(raw_np, layers)

    max_diff = float(np.abs(py_embeds - np_embeds).max())
    logger.info("Projection validation — max abs diff: %.2e", max_diff)
    if max_diff >= 1e-4:
        raise AssertionError(
            f"NumPy projection mismatch (max diff = {max_diff:.2e}). "
            "Check that the layer extraction order is correct."
        )
    logger.info("Validation passed.")


def encode_demo_tracks(
    model: ContrastiveModel,
    audio_dir: Path,
    tracks: list[dict],
    config,
    device: str,
) -> list[list[float]]:
    """Encode each demo track and return audio embeddings as nested lists."""
    spectrograms = []
    for track in tracks:
        audio_path = audio_dir / track["filename"]
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}. "
                "See manual_download.md for download instructions."
            )
        logger.info("  Processing: %s", audio_path.name)
        spec = audio_to_mel_spectrogram(
            audio_path,
            n_mels=config.n_mels,
            sr=config.sample_rate,
            max_seconds=config.max_audio_seconds,
        )
        spectrograms.append(spec)

    batch = torch.stack(spectrograms).to(device)

    with torch.inference_mode():
        audio_raw = encode_audio(
            model.audio_encoder,
            model.audio_extractor,
            batch,
        )
        audio_embeds = model.audio_projection_head(audio_raw)

    return audio_embeds.cpu().float().tolist()


def load_tracks_csv(csv_path: Path) -> list[dict]:
    """Load track metadata from CSV, assigning sequential ids."""
    tracks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tracks.append(
                {
                    "id": i,
                    "title": row["title"],
                    "composer": row["composer"],
                    "filename": row["filename"],
                    "description": row["description"],
                    "era": row.get("era", ""),
                }
            )
    return tracks


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Export web demo data (text projection weights + audio embeddings)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (e.g. checkpoints/best.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Config TOML (default: configs/default.toml)",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing the 15 demo MP3 files",
    )
    parser.add_argument(
        "--tracks-csv",
        required=True,
        help="CSV file with columns: title,composer,filename,description,era",
    )
    parser.add_argument(
        "--output-dir",
        default="web/static/data",
        help="Output directory (default: web/static/data)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = config.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and load model
    logger.info("Building model on %s...", device)
    model = build_model(config)
    ckpt = torch.load(
        args.checkpoint, weights_only=False, map_location=device
    )
    cleaned = _strip_compile_prefix(ckpt["model"])
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    # Export text projection weights
    logger.info("Extracting text projection head weights...")
    layers = extract_projection_layers(model.text_projection_head)
    validate_text_projection(model, layers)
    weights_payload = {"layers": layers}

    out = output_dir / "text_projection.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f)
    logger.info("Saved %s (%.1f KB)", out, out.stat().st_size / 1024)

    # Load track metadata
    tracks = load_tracks_csv(Path(args.tracks_csv))
    logger.info("Loaded %d tracks from %s", len(tracks), args.tracks_csv)

    # Encode audio tracks
    logger.info("Encoding %d audio tracks (this may take a minute)...", len(tracks))
    embeddings = encode_demo_tracks(
        model, Path(args.audio_dir), tracks, config, device
    )

    out = output_dir / "audio_embeddings.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    logger.info("Saved %s (%.1f KB)", out, out.stat().st_size / 1024)

    # Save track metadata (strip filename from public manifest)
    out = output_dir / "tracks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)
    logger.info("Saved %s", out)

    logger.info("Export complete. Output in %s/", output_dir)


if __name__ == "__main__":
    main()
