"""Load and freeze pretrained text and audio encoders."""

from __future__ import annotations

from typing import Union

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import (
    ASTFeatureExtractor,
    ASTModel,
    HubertModel,
    Wav2Vec2FeatureExtractor,
)

AudioModel = Union[ASTModel, HubertModel]
AudioExtractor = Union[ASTFeatureExtractor, Wav2Vec2FeatureExtractor]

MERT_MODELS = {
    "m-a-p/MERT-v1-330M",
    "m-a-p/MERT-v1-95M",
}


def is_mert(name: str) -> bool:
    """Check if a model name is a known MERT model."""
    return name in MERT_MODELS


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
) -> tuple[AudioModel, AudioExtractor, int]:
    """Load a frozen audio encoder (AST or MERT).

    The model is set to eval mode with all parameters frozen.
    Returns the model, its feature extractor, and the output
    embedding dimension.

    Args:
        name: HuggingFace model name. Known MERT models are
            loaded via HubertModel; everything else as AST.
        device: Device to load the model onto.

    Returns:
        ``(model, feature_extractor, embed_dim)``: the frozen
        encoder, its preprocessor, and output dimensionality.
    """
    if is_mert(name):
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        model = HubertModel.from_pretrained(name)
        model.to(device)  # type: ignore[invalid-argument-type]
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model, extractor, model.config.hidden_size

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
    *,
    training: bool = False,
) -> Tensor:
    """Encode a list of strings into embeddings.

    Args:
        model: A frozen or partially unfrozen SentenceTransformer.
        texts: List of input strings.
        training: If True, use tokenize + forward directly to allow
            gradient flow (bypasses SentenceTransformer.encode's
            internal no_grad). When False, uses inference_mode.

    Returns:
        ``(N, embed_dim)`` tensor of text embeddings.
    """
    if training:
        features = model.tokenize(texts)
        features = {k: v.to(model.device) for k, v in features.items()}
        output = model.forward(features)
        return output["sentence_embedding"]
    with torch.inference_mode():
        return model.encode(texts, convert_to_tensor=True)


def unfreeze_text_top_layers(
    model: SentenceTransformer,
    num_layers: int,
) -> None:
    """Unfreeze the top N transformer layers of the text encoder.

    SentenceTransformer wraps a HuggingFace model; transformer layers
    are at ``model[0].auto_model.encoder.layer``. Unfreezes the top
    ``num_layers`` plus the pooler, setting them to train mode.

    Args:
        model: The SentenceTransformer (initially fully frozen).
        num_layers: Number of top encoder layers to unfreeze.
            If 0, nothing changes.
    """
    if num_layers <= 0:
        return
    encoder_layers = model[0].auto_model.encoder.layer
    total = len(encoder_layers)
    if num_layers > total:
        raise ValueError(
            f"Cannot unfreeze {num_layers} layers; "
            f"model has only {total} encoder layers"
        )
    for layer in encoder_layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
        layer.train()
    pooler = model[0].auto_model.pooler
    if pooler is not None:
        for param in pooler.parameters():
            param.requires_grad = True


def _get_audio_encoder_layers(model: AudioModel) -> torch.nn.ModuleList:
    """Return the transformer layer list for an audio encoder."""
    if isinstance(model, HubertModel):
        return model.encoder.layers
    return model.encoder.layer


def unfreeze_audio_top_layers(
    model: AudioModel,
    num_layers: int,
) -> None:
    """Unfreeze the top N transformer layers of the audio encoder.

    Supports both AST (``model.encoder.layer``) and MERT/Hubert
    (``model.encoder.layers``). Also unfreezes the final layer
    norm. The rest of the model stays frozen in eval mode.

    Args:
        model: The audio encoder (AST or MERT).
        num_layers: Number of top encoder layers to unfreeze.
            If 0, nothing changes.
    """
    if num_layers <= 0:
        return
    layers = _get_audio_encoder_layers(model)
    total = len(layers)
    if num_layers > total:
        raise ValueError(
            f"Cannot unfreeze {num_layers} layers; "
            f"model has only {total} encoder layers"
        )
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
        layer.train()
    # Both AST and MERT have a final layer norm to unfreeze.
    if isinstance(model, HubertModel):
        norm = model.encoder.layer_norm
    else:
        norm = model.layernorm
    for param in norm.parameters():
        param.requires_grad = True


def encode_audio(
    model: AudioModel,
    extractor: AudioExtractor,  # noqa: ARG001
    audio_input: Tensor,
    *,
    training: bool = False,
) -> Tensor:
    """Encode audio into embeddings via AST or MERT.

    For AST, ``audio_input`` is ``(N, n_mels, time_frames)`` mel
    spectrograms and the CLS token embedding is returned.

    For MERT (HubertModel), ``audio_input`` is ``(N, samples)``
    raw waveforms and the output is mean-pooled over time.

    Args:
        model: An audio encoder (AST or MERT).
        extractor: The corresponding feature extractor.
        audio_input: Spectrograms for AST, waveforms for MERT.
        training: If True, skip ``torch.inference_mode()`` so
            gradients can flow through unfrozen layers.

    Returns:
        ``(N, embed_dim)`` tensor of audio embeddings.
    """
    use_mert = isinstance(model, HubertModel)
    if training:
        output = model(input_values=audio_input)
        if use_mert:
            return output.last_hidden_state.mean(dim=1)
        return output.last_hidden_state[:, 0, :]
    with torch.inference_mode():
        output = model(input_values=audio_input)
        if use_mert:
            return output.last_hidden_state.mean(dim=1)
        return output.last_hidden_state[:, 0, :]
