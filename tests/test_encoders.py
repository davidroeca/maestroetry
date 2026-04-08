"""Tests for the CLAP encoder helper functions.

These tests validate function contracts only; they do not load
the real CLAP model (which would require a network download).
"""

from __future__ import annotations

import inspect

from maestroetry import encoders


def test_load_clap_model_signature():
    sig = inspect.signature(encoders.load_clap_model)
    params = list(sig.parameters.keys())
    assert "model_name" in params
    assert "device" in params


def test_unfreeze_audio_signature():
    sig = inspect.signature(encoders.unfreeze_clap_audio_top_layers)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "n" in params


def test_unfreeze_text_signature():
    sig = inspect.signature(encoders.unfreeze_clap_text_top_layers)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "n" in params


def test_encode_text_signature():
    sig = inspect.signature(encoders.encode_text)
    params = list(sig.parameters.keys())
    assert "texts" in params
    assert "model" in params
    assert "processor" in params
    assert "training" in params


def test_encode_audio_signature():
    sig = inspect.signature(encoders.encode_audio)
    params = list(sig.parameters.keys())
    assert "waveforms" in params
    assert "model" in params
    assert "processor" in params
    assert "training" in params
