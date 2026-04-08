"""Tests for training configuration."""

from __future__ import annotations

import tempfile

from maestroetry.config import TrainConfig, load_config


def test_default_config_values():
    """TrainConfig defaults match the CLAP fine-tuning recipe."""
    cfg = TrainConfig()
    assert cfg.clap_model_name == "laion/larger_clap_music"
    assert cfg.unfreeze_audio_layers == 2
    assert cfg.unfreeze_text_layers == 2
    assert cfg.encoder_learning_rate == 1e-6
    assert cfg.temperature_init == 0.07
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 3e-4
    assert cfg.max_audio_seconds == 10.0


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
unfreeze_audio_layers = 4
"""
    with tempfile.NamedTemporaryFile(suffix=".toml") as f:
        f.write(toml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.batch_size == 128
    assert cfg.learning_rate == 1e-3
    assert cfg.unfreeze_audio_layers == 4
    # Unspecified values keep defaults
    assert cfg.temperature_init == 0.07


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
