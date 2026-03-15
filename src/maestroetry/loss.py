"""InfoNCE contrastive loss for cross-modal alignment."""

from __future__ import annotations

from torch import Tensor


def info_nce_loss(
    text_embeds: Tensor,
    audio_embeds: Tensor,
    temperature: float | Tensor = 0.07,
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

    With random unit embeddings the expected loss is approximately
    ``log(batch_size)``.

    Args:
        text_embeds: (N, D) L2-normalized text embeddings.
        audio_embeds: (N, D) L2-normalized audio embeddings.
        temperature: Scalar temperature τ controlling softmax
            sharpness. Can be a learnable ``Tensor``.

    Returns:
        Scalar loss tensor (mean of both directions).
    """
    raise NotImplementedError
