"""Export CLAP audio + prompt embeddings for the web demo.

Loads the trained CLAP-based ContrastiveModel and pre-computes
512-dim embeddings for the demo audio tracks plus a fixed vocabulary
of prompt chips. The browser uses these embeddings directly: no
in-browser text encoder is loaded (CLAP is not yet available in
Transformers.js).

Outputs in web/static/data/:
  - audio_embeddings.json   (15 x 512)
  - prompt_embeddings.json  (chip text -> 512-d vector)
  - tracks.json             (track metadata manifest)

Usage:
    uv run export_web_data.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/default.toml \\
        --audio-dir demo_audio/ \\
        --tracks-csv demo_tracks.csv \\
        --output-dir web/static/data/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch

from eval_checkpoints import _strip_compile_prefix, build_model
from maestroetry.config import load_config
from maestroetry.dataset import audio_to_waveform
from maestroetry.encoders import encode_audio, encode_text
from maestroetry.model import ContrastiveModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Fixed vocabulary of prompt chips (must match web/src/routes/+page.svelte).
PROMPT_CHIPS = [
    "A smooth and relaxed jazz piece with soft piano, brushed drums, and a warm inviting feel",
    "An aggressive rock track with distorted electric guitars and pounding, driving drums",
    "A deeply emotional piano melody with gentle strings, conveying sadness and longing",
    "A high-energy electronic dance track with pulsing synthesizers and four-on-the-floor beats",
    "A groovy funk song with tight bass, syncopated guitar riffs, and punchy brass stabs",
    "A dark and unsettling piece with eerie string drones and slow creeping tension",
    "A lively Celtic folk tune with fiddle melodies and rhythmic, joyful energy",
    "A grand orchestral composition with triumphant brass, soaring strings, and powerful timpani",
]

_CLAP_SR = 48000


def encode_demo_tracks(
    model: ContrastiveModel,
    audio_dir: Path,
    tracks: list[dict],
    max_seconds: float,
    device: str,
) -> list[list[float]]:
    """Encode each demo track into a 512-dim CLAP audio embedding."""
    waveforms = []
    for track in tracks:
        audio_path = audio_dir / track["filename"]
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}. "
                "See manual_download.md for download instructions."
            )
        logger.info("  Processing: %s", audio_path.name)
        wav = audio_to_waveform(
            audio_path, sr=_CLAP_SR, max_seconds=max_seconds
        )
        waveforms.append(wav)

    batch = torch.stack(waveforms).to(device)
    audio_embeds = encode_audio(batch, model.clap, model.processor)
    return audio_embeds.cpu().float().tolist()


def encode_prompt_chips(
    model: ContrastiveModel,
    prompts: list[str],
) -> dict[str, list[float]]:
    """Encode the fixed prompt vocabulary into a {text: vector} dict."""
    embeds = encode_text(prompts, model.clap, model.processor)
    embeds_list = embeds.cpu().float().tolist()
    return {text: vec for text, vec in zip(prompts, embeds_list)}


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
        description="Export CLAP audio + prompt embeddings for the web demo"
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

    logger.info("Building model on %s...", device)
    model = build_model(config)
    ckpt = torch.load(
        args.checkpoint, weights_only=False, map_location=device
    )
    cleaned = _strip_compile_prefix(ckpt["model"])
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    tracks = load_tracks_csv(Path(args.tracks_csv))
    logger.info("Loaded %d tracks from %s", len(tracks), args.tracks_csv)

    logger.info(
        "Encoding %d audio tracks (this may take a minute)...", len(tracks)
    )
    audio_embeddings = encode_demo_tracks(
        model,
        Path(args.audio_dir),
        tracks,
        max_seconds=config.max_audio_seconds,
        device=device,
    )

    out = output_dir / "audio_embeddings.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audio_embeddings, f)
    logger.info("Saved %s (%.1f KB)", out, out.stat().st_size / 1024)

    logger.info("Encoding %d prompt chips...", len(PROMPT_CHIPS))
    prompt_embeddings = encode_prompt_chips(model, PROMPT_CHIPS)
    out = output_dir / "prompt_embeddings.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prompt_embeddings, f, indent=2)
    logger.info("Saved %s (%.1f KB)", out, out.stat().st_size / 1024)

    out = output_dir / "tracks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)
    logger.info("Saved %s", out)

    logger.info("Export complete. Output in %s/", output_dir)


if __name__ == "__main__":
    main()
