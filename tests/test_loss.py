"""Tests for InfoNCE loss."""

from __future__ import annotations

import inspect

import torch
from maestroetry.loss import info_nce_loss


def test_info_nce_loss_returns_scalar():
    """Loss function returns a scalar tensor."""
    text = torch.randn(8, 256)
    audio = torch.randn(8, 256)
    loss = info_nce_loss(text, audio)
    assert loss.shape == ()
    assert loss.item() > 0


def test_info_nce_loss_signature():
    """Loss function accepts the expected arguments."""

    sig = inspect.signature(info_nce_loss)
    params = list(sig.parameters.keys())
    assert "text_embeds" in params
    assert "audio_embeds" in params
    assert "temperature" in params
