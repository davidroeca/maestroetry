"""Tests for InfoNCE loss."""

from __future__ import annotations


import pytest
import torch

from maestroetry.loss import info_nce_loss


def test_info_nce_loss_raises_not_implemented():
    """Loss function is a stub that raises NotImplementedError."""
    text = torch.randn(8, 256)
    audio = torch.randn(8, 256)
    with pytest.raises(NotImplementedError):
        info_nce_loss(text, audio)


def test_info_nce_loss_signature():
    """Loss function accepts the expected arguments."""
    import inspect

    sig = inspect.signature(info_nce_loss)
    params = list(sig.parameters.keys())
    assert "text_embeds" in params
    assert "audio_embeds" in params
    assert "temperature" in params
