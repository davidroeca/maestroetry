"""Tests for projection head module."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from maestroetry.projection import (
    ContrastiveModel,
    ProjectionHead,
    get_trainable_params,
)


def test_projection_head_output_shape():
    """ProjectionHead produces correctly shaped, normalized output."""

    head = ProjectionHead(d_in=384)
    x = torch.randn(4, 384)
    out = head(x)
    assert out.shape == (4, 256)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_projection_head_class_exists():
    """ProjectionHead is an nn.Module subclass."""

    assert issubclass(ProjectionHead, nn.Module)


def test_contrastive_model_class_exists():
    """ContrastiveModel is an nn.Module subclass."""

    assert issubclass(ContrastiveModel, nn.Module)


def test_get_trainable_params_signature():
    """get_trainable_params has the expected signature."""

    sig = inspect.signature(get_trainable_params)
    params = list(sig.parameters.keys())
    assert "model" in params
