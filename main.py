"""CLI entrypoint for maestroetry."""

from __future__ import annotations

import argparse
import logging


def _cmd_train(args: argparse.Namespace) -> None:
    from maestroetry.config import load_config
    from maestroetry.train import train

    config = load_config(args.config)
    train(config)


def _cmd_cache_spectrograms(args: argparse.Namespace) -> None:
    from maestroetry.dataset import cache_spectrograms

    cache_spectrograms(args.audio_dir, args.cache_dir)


def _cmd_ingest_musiccaps(args: argparse.Namespace) -> None:
    from maestroetry.ingest import ingest_musiccaps

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest_musiccaps(data_dir=args.data_dir, max_samples=args.max_samples)


def main() -> None:
    """Dispatch CLI commands."""
    parser = argparse.ArgumentParser(
        prog="maestroetry",
        description="Contrastive text-audio retrieval model",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to TOML config file (default: configs/default.toml)",
    )
    p_train.set_defaults(func=_cmd_train)

    # cache-spectrograms
    p_cache = subparsers.add_parser(
        "cache-spectrograms",
        help="Pre-compute mel spectrograms",
    )
    p_cache.add_argument(
        "--audio-dir",
        default="data/audio",
        help="Directory containing audio files (default: data/audio)",
    )
    p_cache.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Output directory for cached tensors (default: data/cache)",
    )
    p_cache.set_defaults(func=_cmd_cache_spectrograms)

    # ingest-musiccaps
    p_ingest = subparsers.add_parser(
        "ingest-musiccaps",
        help="Download MusicCaps dataset",
    )
    p_ingest.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data)",
    )
    p_ingest.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to download",
    )
    p_ingest.set_defaults(func=_cmd_ingest_musiccaps)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
