"""Evaluation metrics for cross-modal retrieval."""

from __future__ import annotations

from torch import Tensor


def recall_at_k(
    text_embeds: Tensor,
    audio_embeds: Tensor,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute Recall@k for text-audio and audio-text retrieval.

    Given N paired embeddings, for each text query rank all audio
    embeddings by cosine similarity and check whether the correct
    match appears in the top k. Repeat in the audio-to-text direction.

    Embeddings are assumed to be L2-normalized.

    Args:
        text_embeds: ``(N, D)`` L2-normalized text embeddings.
        audio_embeds: ``(N, D)`` L2-normalized audio embeddings.
        k_values: List of k values to evaluate. Defaults to
            ``[1, 5, 10]``.

    Returns:
        Dict with keys like ``"t2a_R@1"``, ``"t2a_R@5"``,
        ``"a2t_R@1"``, etc., with float values in ``[0, 1]``.
    """
    raise NotImplementedError
