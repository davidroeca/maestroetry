"""Evaluation metrics for cross-modal retrieval."""

from __future__ import annotations

import torch
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
    if k_values is None:
        k_values = [1, 5, 10]
    t2a_sim = text_embeds @ audio_embeds.T
    a2t_sim = t2a_sim.T

    n = t2a_sim.size(0)
    targets = torch.arange(n, device=text_embeds.device)
    t2a_ranked = torch.argsort(t2a_sim, dim=1, descending=True)
    a2t_ranked = torch.argsort(a2t_sim, dim=1, descending=True)

    result: dict[str, float] = {}
    for k in k_values:
        top_k_t2a_indices = t2a_ranked[:, :k]
        t2a_hits = (top_k_t2a_indices == targets.unsqueeze(1)).any(dim=1)
        result[f"t2a_R@{k}"] = t2a_hits.float().mean().item()

        top_k_a2t_indices = a2t_ranked[:, :k]
        a2t_hits = (top_k_a2t_indices == targets.unsqueeze(1)).any(dim=1)
        result[f"a2t_R@{k}"] = a2t_hits.float().mean().item()

    return result
