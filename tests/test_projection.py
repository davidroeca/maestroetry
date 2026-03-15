"""Tests for projection head module."""

from __future__ import annotations

import pytest

from maestroetry.projection import (
    ContrastiveModel,
    ProjectionHead,
    get_trainable_params,
)


def test_projection_head_raises_not_implemented():
    """ProjectionHead.__init__ is a stub."""
    with pytest.raises(NotImplementedError):
        ProjectionHead(d_in=384)


def test_projection_head_class_exists():
    """ProjectionHead is an nn.Module subclass."""
    import torch.nn as nn

    assert issubclass(ProjectionHead, nn.Module)


def test_contrastive_model_class_exists():
    """ContrastiveModel is an nn.Module subclass."""
    import torch.nn as nn

    assert issubclass(ContrastiveModel, nn.Module)


def test_get_trainable_params_signature():
    """get_trainable_params has the expected signature."""
    import inspect

    sig = inspect.signature(get_trainable_params)
    params = list(sig.parameters.keys())
    assert "model" in params
