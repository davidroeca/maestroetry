"""CLI entrypoint for maestroetry."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch CLI commands: train, eval, cache-spectrograms."""
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [options]")
        print("Commands: train, eval, cache-spectrograms")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/default.toml"
        from maestroetry.config import load_config
        from maestroetry.train import train

        config = load_config(config_path)
        train(config)

    elif command == "eval":
        raise NotImplementedError("eval command not yet implemented")

    elif command == "cache-spectrograms":
        from maestroetry.dataset import cache_spectrograms

        audio_dir = sys.argv[2] if len(sys.argv) > 2 else "data/audio"
        cache_dir = sys.argv[3] if len(sys.argv) > 3 else "data/cache"
        cache_spectrograms(audio_dir, cache_dir)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
