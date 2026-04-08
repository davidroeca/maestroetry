"""Contrastive model wrapping a CLAP backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import ClapModel, ClapProcessor

from maestroetry import encoders


class ContrastiveModel(nn.Module):
    """Thin wrapper around a CLAP model + a learnable temperature.

    CLAP already produces L2-normalized 512-dim embeddings in a
    shared text-audio space, so no projection heads are needed.
    """

    def __init__(
        self,
        clap: ClapModel,
        processor: ClapProcessor,
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.clap = clap
        self.processor = processor
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )

    def forward(
        self,
        texts: list[str],
        waveforms: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode text and audio, return embeds plus temperature.

        Args:
            texts: List of N text strings.
            waveforms: ``(N, samples)`` 48 kHz waveforms.

        Returns:
            ``(text_embeds (N,512), audio_embeds (N,512), temperature)``.
        """
        text_embeds = encoders.encode_text(
            texts,
            self.clap,
            self.processor,
            training=self.training,
        )
        audio_embeds = encoders.encode_audio(
            waveforms,
            self.clap,
            self.processor,
            training=self.training,
        )
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return text_embeds, audio_embeds, temperature


def get_trainable_params(
    model: ContrastiveModel,
    encoder_lr: float,
    main_lr: float,
) -> list[dict[str, object]]:
    """Return AdamW param groups for the trainable parameters.

    Two groups:
      - unfrozen CLAP layers at ``encoder_lr``
      - the ``log_temperature`` scalar at ``main_lr``

    Args:
        model: The ContrastiveModel.
        encoder_lr: Learning rate for unfrozen CLAP layers.
        main_lr: Learning rate for the temperature scalar.

    Returns:
        Param groups suitable for ``torch.optim.AdamW``.
    """
    encoder_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and name != "log_temperature"
    ]
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": [model.log_temperature], "lr": main_lr},
    ]
