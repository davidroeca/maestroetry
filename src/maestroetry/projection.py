"""Projection heads and contrastive model."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from maestroetry import encoders

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import ASTFeatureExtractor, ASTModel


class ProjectionHead(nn.Module):
    """MLP that projects encoder output into the shared space.

    Configurable-depth MLP with optional dropout. Default (depth=3,
    dropout=0.1) adds one hidden-to-hidden block between input and
    output layers.

    The output is always L2-normalized to lie on the unit hypersphere,
    so dot products equal cosine similarity.

    Layer naming preserves checkpoint compatibility: the first layer
    is always ``linear1``, the final layer is always ``linear2``, and
    any intermediate layers use ``linear_mid*`` names.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 512,
        d_out: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = [
            ("linear1", nn.Linear(d_in, d_hidden)),
            ("relu", nn.ReLU()),
        ]
        if dropout > 0.0:
            layers.append(("dropout1", nn.Dropout(dropout)))
        for i in range(depth - 2):
            layers.append((f"linear_mid{i + 1}", nn.Linear(d_hidden, d_hidden)))
            layers.append((f"relu_mid{i + 1}", nn.ReLU()))
            if dropout > 0.0:
                layers.append((f"dropout_mid{i + 1}", nn.Dropout(dropout)))
        layers.append(("linear2", nn.Linear(d_hidden, d_out)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalize.

        Args:
            x: ``(N, d_in)`` encoder output.

        Returns:
            ``(N, d_out)`` unit-normalized embeddings.
        """
        forward_pass = self.net(x)
        return F.normalize(forward_pass, dim=-1)


class ContrastiveModel(nn.Module):
    """Full contrastive model wiring encoders + projection heads.

    Combines frozen text and audio encoders with trainable
    projection heads and a learnable temperature parameter.
    """

    def __init__(
        self,
        text_encoder: SentenceTransformer,
        audio_encoder: ASTModel,
        audio_extractor: ASTFeatureExtractor,
        text_embed_dim: int,
        audio_embed_dim: int,
        projection_hidden: int = 512,
        projection_out: int = 256,
        projection_depth: int = 3,
        projection_dropout: float = 0.1,
        temperature_init: float = 0.07,
        finetune_audio: bool = False,
        finetune_text: bool = False,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.audio_extractor = audio_extractor
        self.finetune_audio = finetune_audio
        self.finetune_text = finetune_text
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )
        self.text_projection_head = ProjectionHead(
            d_in=text_embed_dim,
            d_hidden=projection_hidden,
            d_out=projection_out,
            depth=projection_depth,
            dropout=projection_dropout,
        )
        self.audio_projection_head = ProjectionHead(
            d_in=audio_embed_dim,
            d_hidden=projection_hidden,
            d_out=projection_out,
            depth=projection_depth,
            dropout=projection_dropout,
        )

    def forward(
        self,
        texts: list[str],
        spectrograms: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode and project both modalities.

        Args:
            texts: List of N text strings.
            spectrograms: ``(N, n_mels, time_frames)`` audio input.

        Returns:
            ``(text_embeds, audio_embeds, temperature)`` where
            embeds are ``(N, projection_out)`` and temperature
            is a positive scalar tensor.
        """
        text_embeds = encoders.encode_text(
            self.text_encoder,
            texts,
            training=self.finetune_text and self.training,
        )
        if not self.finetune_text:
            text_embeds = text_embeds.detach()
        audio_embeds = encoders.encode_audio(
            self.audio_encoder,
            self.audio_extractor,
            spectrograms,
            training=self.finetune_audio and self.training,
        )
        if not self.finetune_audio:
            audio_embeds = audio_embeds.detach()
        text_embeds = self.text_projection_head(text_embeds)
        audio_embeds = self.audio_projection_head(audio_embeds)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return text_embeds, audio_embeds, temperature


def get_trainable_params(
    model: ContrastiveModel,
    encoder_lr: float | None = None,
) -> list[nn.Parameter] | list[dict[str, object]]:
    """Return only the trainable parameters of the model.

    When ``encoder_lr`` is provided and the model is fine-tuning
    the audio encoder, returns two param groups with differential
    learning rates. Otherwise returns a flat parameter list.

    Args:
        model: The ContrastiveModel.
        encoder_lr: Learning rate for unfrozen encoder layers.
            If None or model is not fine-tuning, returns a flat
            list of all trainable parameters.

    Returns:
        Parameter list or param groups suitable for an optimizer.
    """
    if encoder_lr is not None and (model.finetune_audio or model.finetune_text):
        encoder_params: list[nn.Parameter] = []
        if model.finetune_audio:
            encoder_params.extend(
                p for p in model.audio_encoder.parameters() if p.requires_grad
            )
        if model.finetune_text:
            encoder_params.extend(
                p for p in model.text_encoder.parameters() if p.requires_grad
            )
        projection_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad
            and "audio_encoder" not in name
            and "text_encoder" not in name
        ]
        return [
            {"params": projection_params},
            {"params": encoder_params, "lr": encoder_lr},
        ]
    return [p for p in model.parameters() if p.requires_grad]
