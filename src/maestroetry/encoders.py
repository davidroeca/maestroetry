"""Load and partially unfreeze a CLAP model for text-audio retrieval."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import ClapModel, ClapProcessor


def load_clap_model(
    model_name: str = "laion/larger_clap_music",
    device: str = "cpu",
) -> tuple[ClapModel, ClapProcessor]:
    """Load a CLAP model and processor, fully frozen and in eval mode.

    Args:
        model_name: HuggingFace model ID for a CLAP checkpoint.
        device: Device to load the model onto.

    Returns:
        ``(model, processor)`` with all parameters frozen.
    """
    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)
    model.to(device)  # ty: ignore[invalid-argument-type]
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, processor


def _audio_encoder_layers(model: ClapModel) -> list[torch.nn.Module]:
    """Return the flat list of HTSAT swin-transformer blocks.

    CLAP's audio encoder is an HTSAT model whose ``layers`` attribute
    holds a sequence of swin-transformer stages. Each stage contains
    a ``blocks`` ModuleList. We flatten across stages so unfreezing
    "top N" actually walks the deepest blocks first.
    """
    flat: list[torch.nn.Module] = []
    for stage in model.audio_model.audio_encoder.layers:
        for block in stage.blocks:  # ty: ignore[not-iterable]
            flat.append(block)  # ty: ignore[invalid-argument-type]
    return flat


def unfreeze_clap_audio_top_layers(model: ClapModel, n: int) -> None:
    """Unfreeze the top ``n`` HTSAT blocks of the CLAP audio encoder.

    Also unfreezes the audio projection layer that maps the encoder
    output into the shared embedding space.
    """
    if n <= 0:
        return
    blocks = _audio_encoder_layers(model)
    total = len(blocks)
    if n > total:
        raise ValueError(
            f"Cannot unfreeze {n} audio blocks; CLAP audio encoder has {total}"
        )
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
        block.train()
    for param in model.audio_projection.parameters():
        param.requires_grad = True


def unfreeze_clap_text_top_layers(model: ClapModel, n: int) -> None:
    """Unfreeze the top ``n`` RoBERTa layers of the CLAP text encoder.

    Also unfreezes the text projection layer that maps the encoder
    output into the shared embedding space.
    """
    if n <= 0:
        return
    layers = model.text_model.encoder.layer
    total = len(layers)
    if n > total:
        raise ValueError(
            f"Cannot unfreeze {n} text layers; CLAP text encoder has {total}"
        )
    for layer in layers[-n:]:  # ty: ignore[not-iterable]
        for param in layer.parameters():
            param.requires_grad = True
        layer.train()
    for param in model.text_projection.parameters():
        param.requires_grad = True


CLAP_TEXT_INPUT_KEYS = frozenset({"input_ids", "attention_mask", "position_ids"})
CLAP_AUDIO_INPUT_KEYS = frozenset(
    {"input_features", "is_longer", "attention_mask"}
)


def encode_text_from_inputs(
    text_inputs: dict[str, Tensor],
    model: ClapModel,
    *,
    training: bool = False,
) -> Tensor:
    """Encode pre-processed text inputs into 512-dim CLAP embeddings.

    Args:
        text_inputs: Dict of tensors produced by ``ClapProcessor``
            (``input_ids``, ``attention_mask``, optionally
            ``position_ids``).
        model: A CLAP model.
        training: If True, run with gradient tracking.

    Returns:
        ``(N, 512)`` L2-normalized text embeddings.
    """
    device = next(model.parameters()).device
    moved = {
        k: v.to(device, non_blocking=True) for k, v in text_inputs.items()
    }
    if training:
        outputs = model.get_text_features(**moved)
    else:
        with torch.inference_mode():
            outputs = model.get_text_features(**moved)
    return torch.nn.functional.normalize(outputs.pooler_output, dim=-1)


def encode_audio_from_inputs(
    audio_inputs: dict[str, Tensor],
    model: ClapModel,
    *,
    training: bool = False,
) -> Tensor:
    """Encode pre-processed audio inputs into 512-dim CLAP embeddings.

    Args:
        audio_inputs: Dict of tensors produced by ``ClapProcessor``
            (``input_features``, ``is_longer``, ``attention_mask``).
        model: A CLAP model.
        training: If True, run with gradient tracking.

    Returns:
        ``(N, 512)`` L2-normalized audio embeddings.
    """
    device = next(model.parameters()).device
    moved = {
        k: v.to(device, non_blocking=True) for k, v in audio_inputs.items()
    }
    if training:
        outputs = model.get_audio_features(**moved)
    else:
        with torch.inference_mode():
            outputs = model.get_audio_features(**moved)
    return torch.nn.functional.normalize(outputs.pooler_output, dim=-1)


def encode_text(
    texts: list[str],
    model: ClapModel,
    processor: ClapProcessor,
    *,
    training: bool = False,
) -> Tensor:
    """Encode strings into 512-dim CLAP text embeddings (L2-normalized).

    Args:
        texts: List of input strings.
        model: A CLAP model.
        processor: The matching CLAP processor.
        training: If True, run with gradient tracking so backprop
            can flow through any unfrozen layers.

    Returns:
        ``(N, 512)`` L2-normalized text embeddings.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    text_inputs = {
        k: v for k, v in inputs.items() if k in CLAP_TEXT_INPUT_KEYS
    }
    return encode_text_from_inputs(text_inputs, model, training=training)


def encode_audio(
    waveforms: Tensor,
    model: ClapModel,
    processor: ClapProcessor,
    *,
    training: bool = False,
) -> Tensor:
    """Encode raw waveforms into 512-dim CLAP audio embeddings.

    Args:
        waveforms: ``(N, samples)`` raw waveforms at 48 kHz.
        model: A CLAP model.
        processor: The matching CLAP processor.
        training: If True, run with gradient tracking so backprop
            can flow through any unfrozen layers.

    Returns:
        ``(N, 512)`` L2-normalized audio embeddings.
    """
    # ClapProcessor's audio path expects numpy/list inputs.
    audio_list = [w.detach().cpu().float().numpy() for w in waveforms]
    inputs = processor(
        audio=audio_list,
        sampling_rate=48000,
        return_tensors="pt",
    )
    audio_inputs = {
        k: v for k, v in inputs.items() if k in CLAP_AUDIO_INPUT_KEYS
    }
    return encode_audio_from_inputs(audio_inputs, model, training=training)
