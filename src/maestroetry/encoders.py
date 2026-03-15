"""Load and freeze pretrained text and audio encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def encode_audio(
    model: ASTModel,
    extractor: ASTFeatureExtractor,
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
    raise NotImplementedError
