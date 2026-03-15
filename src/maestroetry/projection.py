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

    Architecture: Linear(d_in, 512) -> ReLU -> Linear(512, 256) -> L2 norm.

    The output is always L2-normalized to lie on the unit hypersphere,
    so dot products equal cosine similarity.
    """

    def __init__(
        self, d_in: int, d_hidden: int = 512, d_out: int = 256
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(d_in, d_hidden)),
                    ("relu", nn.ReLU()),
                    ("linear2", nn.Linear(d_hidden, d_out)),
                ]
            )
        )

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
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.audio_extractor = audio_extractor
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )
        self.text_projection_head = ProjectionHead(
            d_in=text_embed_dim,
            d_hidden=projection_hidden,
            d_out=projection_out,
        )
        self.audio_projection_head = ProjectionHead(
            d_in=audio_embed_dim,
            d_hidden=projection_hidden,
            d_out=projection_out,
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
        text_embeds = encoders.encode_text(self.text_encoder, texts)
        audio_embeds = encoders.encode_audio(
            self.audio_encoder,
            self.audio_extractor,
            spectrograms,
        )
        text_embeds = self.text_projection_head(text_embeds)
        audio_embeds = self.audio_projection_head(audio_embeds)
        return text_embeds, audio_embeds, self.log_temperature.exp()


def get_trainable_params(
    model: ContrastiveModel,
) -> list[nn.Parameter]:
    """Return only the trainable parameters of the model.

    This includes projection head weights and the temperature
    parameter, not the frozen encoder weights.
    """
    return [p for p in model.parameters() if p.requires_grad]
