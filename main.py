"""CLI entrypoint for maestroetry."""

from __future__ import annotations

import argparse
import logging


def _cmd_train(args: argparse.Namespace) -> None:
    from maestroetry.config import load_config
    from maestroetry.train import train

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    overrides: dict[str, str] = {}
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    config = load_config(args.config, **overrides)
    train(config, resume=args.resume, weights_only=args.weights_only)


def _cmd_cache_spectrograms(args: argparse.Namespace) -> None:
    from maestroetry.dataset import cache_spectrograms

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache_spectrograms(args.audio_dir, args.cache_dir)


def _cmd_ingest(args: argparse.Namespace) -> None:
    from maestroetry.ingest import ingest

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest(
        data_dir=args.data_dir,
        max_samples_lp=args.max_samples_lp,
        max_samples_jamendo=args.max_samples_jamendo,
        lp_only=args.lp_only,
        jamendo_only=args.jamendo_only,
        lp_all_captions=args.lp_all_captions,
    )


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
    p_train.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for saving checkpoints (overrides config)",
    )
    p_train.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    p_train.add_argument(
        "--weights-only",
        action="store_true",
        help="Load only model weights from checkpoint, skip optimizer/scheduler",
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

    # ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Download LP-MusicCaps + Jamendo datasets",
    )
    p_ingest.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data)",
    )
    p_ingest.add_argument(
        "--max-samples-lp",
        type=int,
        default=None,
        help="Limit number of LP-MusicCaps samples",
    )
    p_ingest.add_argument(
        "--max-samples-jamendo",
        type=int,
        default=None,
        help="Limit number of Jamendo samples",
    )
    p_ingest.add_argument(
        "--lp-only",
        action="store_true",
        help="Only ingest LP-MusicCaps dataset",
    )
    p_ingest.add_argument(
        "--jamendo-only",
        action="store_true",
        help="Only ingest MTG-Jamendo dataset",
    )
    p_ingest.add_argument(
        "--lp-all-captions",
        action="store_true",
        help="Emit all LP-MusicCaps caption variants (4x more pairs)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
