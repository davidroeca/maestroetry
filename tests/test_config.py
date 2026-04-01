"""Tests for training configuration."""

from __future__ import annotations

import tempfile

from maestroetry.config import TrainConfig, load_config


def test_default_config_values():
    """TrainConfig defaults match proposal hyperparameters."""
    cfg = TrainConfig()
    assert cfg.embed_dim == 256
    assert cfg.projection_hidden_dim == 512
    assert cfg.temperature_init == 0.07
    assert cfg.batch_size == 100
    assert cfg.learning_rate == 3e-4
    assert cfg.n_mels == 128
    assert cfg.sample_rate == 16000


def test_config_is_frozen():
    """TrainConfig is immutable."""
    cfg = TrainConfig()
    try:
        cfg.batch_size = 32  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_load_config_from_toml():
    """load_config reads values from a TOML file."""
    toml_content = b"""
[train]
batch_size = 128
learning_rate = 1e-3
embed_dim = 512
"""
    with tempfile.NamedTemporaryFile(suffix=".toml") as f:
        f.write(toml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.batch_size == 128
    assert cfg.learning_rate == 1e-3
    assert cfg.embed_dim == 512
    # Unspecified values keep defaults
    assert cfg.n_mels == 128


def test_load_config_ignores_unknown_keys():
    """Unknown keys in TOML are silently ignored."""
    toml_content = b"""
[train]
batch_size = 32
unknown_key = "should be ignored"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml") as f:
        f.write(toml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.batch_size == 32
