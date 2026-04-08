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


def recall_at_k_by_track(
    text_embeds: Tensor,
    audio_embeds: Tensor,
    track_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute Recall@k against unique tracks instead of caption rows.

    LP-MusicCaps has multiple captions per track, so naive row-indexed
    R@k double-penalizes ties: if rows ``i`` and ``j`` share the same
    audio (because they're two captions for the same track), their
    audio embeddings are identical and a tie-break can never give both
    rows R@1 = 1. The hard ceiling becomes ``1 / captions_per_track``.

    This metric instead deduplicates the audio side by ``track_ids``
    and counts a hit when the retrieved track matches the query's
    track, regardless of which caption row was used. The audio-to-text
    direction is computed once per unique track and counts a hit if
    *any* caption row of that track appears in the top ``k``.

    Args:
        text_embeds: ``(N, D)`` per-row L2-normalized text embeddings.
        audio_embeds: ``(N, D)`` per-row L2-normalized audio embeddings.
            Rows that share a ``track_id`` must have identical audio
            embeddings; only the first occurrence is kept.
        track_ids: Length-``N`` list of track identifiers, one per row,
            in the same order as ``text_embeds`` / ``audio_embeds``.
        k_values: List of k values to evaluate. Defaults to
            ``[1, 5, 10]``.

    Returns:
        Dict with keys like ``"t2a_R@1"``, ``"a2t_R@1"``, etc.
    """
    if k_values is None:
        k_values = [1, 5, 10]
    if len(track_ids) != text_embeds.size(0):
        msg = (
            f"track_ids length {len(track_ids)} does not match "
            f"text_embeds rows {text_embeds.size(0)}"
        )
        raise ValueError(msg)

    track_to_unique: dict[str, int] = {}
    unique_audio_rows: list[int] = []
    for row_idx, tid in enumerate(track_ids):
        if tid not in track_to_unique:
            track_to_unique[tid] = len(unique_audio_rows)
            unique_audio_rows.append(row_idx)
    unique_audio = audio_embeds[torch.tensor(unique_audio_rows)]
    n_text = text_embeds.size(0)
    text_targets = torch.tensor(
        [track_to_unique[tid] for tid in track_ids], dtype=torch.long
    )

    t2a_sim = text_embeds @ unique_audio.T
    t2a_ranked = torch.argsort(t2a_sim, dim=1, descending=True)

    a2t_sim = unique_audio @ text_embeds.T
    a2t_ranked = torch.argsort(a2t_sim, dim=1, descending=True)

    rows_per_track: list[list[int]] = [[] for _ in unique_audio_rows]
    for row_idx, tid in enumerate(track_ids):
        rows_per_track[track_to_unique[tid]].append(row_idx)

    result: dict[str, float] = {}
    for k in k_values:
        top_k_t2a = t2a_ranked[:, :k]
        t2a_hits = (top_k_t2a == text_targets.unsqueeze(1)).any(dim=1)
        result[f"t2a_R@{k}"] = t2a_hits.float().mean().item()

        a2t_hit_count = 0
        top_k_a2t = a2t_ranked[:, :k]
        for u_idx, valid_rows in enumerate(rows_per_track):
            valid = torch.tensor(valid_rows, dtype=torch.long)
            in_top = torch.isin(top_k_a2t[u_idx], valid).any().item()
            if in_top:
                a2t_hit_count += 1
        result[f"a2t_R@{k}"] = a2t_hit_count / max(len(unique_audio_rows), 1)

    return result
