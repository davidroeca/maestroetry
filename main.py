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


def _cmd_ingest(args: argparse.Namespace) -> None:
    from maestroetry.ingest import ingest

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest(
        data_dir=args.data_dir,
        fma_subset=args.fma_subset,
        max_samples_sdd=args.max_samples_sdd,
        max_samples_fma=args.max_samples_fma,
        sdd_only=args.sdd_only,
        fma_only=args.fma_only,
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
        help="Download SDD + FMA datasets",
    )
    p_ingest.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data)",
    )
    p_ingest.add_argument(
        "--fma-subset",
        default="small",
        choices=["small", "medium"],
        help="FMA subset to download (default: small)",
    )
    p_ingest.add_argument(
        "--max-samples-sdd",
        type=int,
        default=None,
        help="Limit number of SDD samples",
    )
    p_ingest.add_argument(
        "--max-samples-fma",
        type=int,
        default=None,
        help="Limit number of FMA samples",
    )
    p_ingest.add_argument(
        "--sdd-only",
        action="store_true",
        help="Only ingest Song Describer Dataset",
    )
    p_ingest.add_argument(
        "--fma-only",
        action="store_true",
        help="Only ingest FMA dataset",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
