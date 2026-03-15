"""Load and freeze pretrained text and audio encoders."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import ASTFeatureExtractor, ASTModel


def load_text_encoder(
    name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> tuple[SentenceTransformer, int]:
    """Load a frozen sentence-transformer text encoder.

    The model is set to eval mode with all parameters frozen
    (``requires_grad=False``). Returns the model and its output
    embedding dimension.

    Args:
        name: HuggingFace sentence-transformers model name.
        device: Device to load the model onto.

    Returns:
        ``(model, embed_dim)``: the frozen encoder and its
        output dimensionality (384 for MiniLM).
    """
    model = SentenceTransformer(name, device=device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    dimensionality = model.get_sentence_embedding_dimension()
    if dimensionality is None:
        raise ValueError(
            f'get_sentence_embedding_dimension() returned None for model "{name}"'
        )
    model.eval()
    return model, dimensionality


def load_audio_encoder(
    name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    device: str = "cpu",
) -> tuple[ASTModel, ASTFeatureExtractor, int]:
    """Load a frozen Audio Spectrogram Transformer encoder.

    The model is set to eval mode with all parameters frozen.
    Returns the model, its feature extractor, and the output
    embedding dimension.

    Args:
        name: HuggingFace AST model name.
        device: Device to load the model onto.

    Returns:
        ``(model, feature_extractor, embed_dim)``: the frozen
        encoder, its preprocessor, and output dimensionality.
    """
    extractor = ASTFeatureExtractor.from_pretrained(name)
    model = ASTModel.from_pretrained(name)
    model.to(device)  # type: ignore[invalid-argument-type]
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, extractor, model.config.hidden_size


def encode_text(
    model: SentenceTransformer,
    texts: list[str],
) -> Tensor:
    """Encode a list of strings into embeddings.

    Args:
        model: A frozen SentenceTransformer.
        texts: List of input strings.

    Returns:
        ``(N, embed_dim)`` tensor of text embeddings.
    """
    # Redundant but explicit no_grad
    with torch.no_grad():
        return model.encode(texts, convert_to_tensor=True)


def encode_audio(
    model: ASTModel,
    extractor: ASTFeatureExtractor,  # noqa: ARG001
    spectrograms: Tensor,
) -> Tensor:
    """Encode mel spectrograms into embeddings via AST.

    Args:
        model: A frozen ASTModel.
        extractor: The corresponding ASTFeatureExtractor.
        spectrograms: ``(N, n_mels, time_frames)`` tensor of
            mel spectrograms.

    Returns:
        ``(N, embed_dim)`` tensor of audio embeddings.
    """
    with torch.no_grad():
        output = model(input_values=spectrograms)
        cls_tokens = output.last_hidden_state[:, 0, :]
        return cls_tokens
