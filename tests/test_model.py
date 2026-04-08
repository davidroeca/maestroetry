"""Tests for the CLAP-based ContrastiveModel wrapper."""

from __future__ import annotations

import inspect

import torch.nn as nn

from maestroetry.model import ContrastiveModel, get_trainable_params


def test_contrastive_model_class_exists():
    """ContrastiveModel is an nn.Module subclass."""
    assert issubclass(ContrastiveModel, nn.Module)


def test_contrastive_model_init_signature():
    """ContrastiveModel takes (clap, processor, temperature_init)."""
    sig = inspect.signature(ContrastiveModel.__init__)
    params = list(sig.parameters.keys())
    assert "clap" in params
    assert "processor" in params
    assert "temperature_init" in params


def test_get_trainable_params_signature():
    """get_trainable_params requires encoder_lr and main_lr."""
    sig = inspect.signature(get_trainable_params)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "encoder_lr" in params
    assert "main_lr" in params
