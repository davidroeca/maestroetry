"""InfoNCE contrastive loss for cross-modal alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(
    text_embeds: Tensor,
    audio_embeds: Tensor,
    temperature: float | Tensor = 0.07,
    hard_negative_weight: float = 1.0,
) -> Tensor:
    """Compute symmetric InfoNCE loss.

    Given a batch of N paired (text, audio) embeddings, compute the
    NxN cosine similarity matrix scaled by temperature. The loss is
    the average of two cross-entropy terms:

    - **text to audio**: for each text embedding, classify which of
      the N audio embeddings is the correct match (target = diagonal).
    - **audio to text**: for each audio embedding, classify which of
      the N text embeddings is the correct match (target = diagonal).

    Embeddings are assumed to be L2-normalized, so dot product equals
    cosine similarity.

    When ``hard_negative_weight`` (beta) is greater than 1, off-diagonal
    logits are scaled up so that hard negatives (high false similarity)
    dominate the loss denominator. ``beta = 1.0`` recovers standard
    InfoNCE with no overhead.

    With random unit embeddings the expected loss is approximately
    ``log(batch_size)``.

    Args:
        text_embeds: (N, D) L2-normalized text embeddings.
        audio_embeds: (N, D) L2-normalized audio embeddings.
        temperature: Scalar temperature τ controlling softmax
            sharpness. Can be a learnable ``Tensor``.
        hard_negative_weight: Concentration parameter beta for
            scaling off-diagonal (negative) logits. 1.0 = standard
            InfoNCE.

    Returns:
        Scalar loss tensor (mean of both directions).
    """
    cos_similarity = text_embeds @ audio_embeds.T
    t2a_logits = cos_similarity / temperature
    if hard_negative_weight != 1.0:
        mask = torch.eye(
            t2a_logits.size(0), device=t2a_logits.device, dtype=torch.bool
        )
        t2a_logits = t2a_logits.clone()
        t2a_logits[~mask] *= hard_negative_weight
    a2t_logits = t2a_logits.T
    t2a_loss = F.cross_entropy(
        t2a_logits,
        torch.arange(t2a_logits.size(0), device=t2a_logits.device),
    )
    a2t_loss = F.cross_entropy(
        a2t_logits,
        torch.arange(a2t_logits.size(0), device=a2t_logits.device),
    )
    return (t2a_loss + a2t_loss) / 2
