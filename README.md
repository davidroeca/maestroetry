# Maestroetry

A contrastive text-audio retrieval model inspired by CLIP. It aligns text (poems, lyrics, descriptions) and audio (songs, clips) in a shared embedding space to enable cross-modal retrieval. Given a poem, find the song that matches its mood. Given a song, find the text that describes it.

This is a first PyTorch training project, built for someone with strong Python experience who wants to learn model training hands-on.

## How it works

Two frozen pretrained encoders extract features from each modality:

- **Text:** `all-MiniLM-L6-v2` (sentence-transformers), producing 384-dim embeddings
- **Audio:** `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer), processing mel spectrograms as image patches via ViT-style attention

Two frozen pretrained **encoders** extract features from each modality and are loaded via `load_text_encoder` and `load_audio_encoder`. Both are set to eval mode with all parameters frozen. `encode_text` uses `SentenceTransformer.encode` to produce `(N, 384)` embeddings. `encode_audio` passes mel spectrograms directly to the AST model and extracts the CLS token from `last_hidden_state[:, 0, :]`, giving `(N, 768)` embeddings. Both encode functions run under `torch.no_grad()`.

Two small trainable **projection heads** (Linear -> ReLU -> Linear -> L2 norm) map both encoder outputs into a shared 256-dimensional space. These are the only parameters that get trained.

Training uses **InfoNCE loss**: for a batch of N paired (text, audio) embeddings, the model computes an NxN cosine similarity matrix (via `text_embeds @ audio_embeds.T`, which equals cosine similarity because both are L2-normalized). The matrix is scaled by `1/temperature` to produce logits. Two cross-entropy terms are computed: one row-wise (each text finds its audio match among N options) and one column-wise (each audio finds its text match). The diagonal of the matrix is the target in both directions. The final loss is their average. Off-diagonal entries act as in-batch negatives, so larger batch sizes produce harder training signal.

## Getting started

### Environment setup

PyTorch is installed as an optional extra so the right build (CPU or CUDA) is
used per environment. Pick the setup for your machine.

**Local dev (CPU-only Linux)**

```bash
uv sync --extra cpu
```

**Google Colab (CUDA 12.8)**

Colab has CUDA drivers but no uv, so install uv and sync explicitly:

```python
!pip install uv -q
!uv sync --extra cu128
```

Then prefix any commands with `!uv run` instead of `uv run`.

### Running the project

```bash
# Cache mel spectrograms from your audio directory
uv run python main.py cache-spectrograms data/audio data/cache

# Train
uv run python main.py train configs/default.toml
```

You'll need a CSV manifest at `data/manifest.csv` with `audio_path` and `text` columns pointing to your (audio, text) pairs.

## What you'll implement

The stubs in `src/maestroetry/` are ready to fill in:

- **`projection.py`**: `ProjectionHead` and `ContrastiveModel` as `nn.Module` subclasses
- **`dataset.py`**: mel spectrogram conversion with librosa, caching, and a PyTorch `Dataset`
- **`train.py`**: training loop with AdamW and linear warmup, plus TensorBoard logging
- **`evaluate.py`**: Recall@k for text-to-audio and audio-to-text retrieval

`config.py` is already implemented: a frozen dataclass with TOML file loading.

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Shared embedding dim | 256 |
| Batch size | 64 |
| Learning rate | 3e-4 (AdamW) |
| Temperature tau | 0.07 (learnable) |
| Mel bins | 128 |
| Max audio length | 10s |

## Data

A few thousand well-curated pairs is enough to get meaningful results. The frozen encoders already provide strong representations, so the projection heads just need to learn the alignment mapping. Some options:

- **Free Music Archive (FMA)**: creative-commons music with genre tags and metadata
- **Lyrics datasets**: pair songs with their lyrics
- **LLM-generated descriptions**: use an LLM to write rich text descriptions of audio clips

## Future direction

Once trained, the model exposes a retrieval interface suitable for LLM tool use:

- `search_by_text(query)`: find audio matching a text description
- `search_by_audio(path)`: find text describing a piece of audio
- `similarity(text, audio_path)`: score a text-audio pair
