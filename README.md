# Maestroetry

Contrastive text-audio retrieval model inspired by CLIP. Aligns text (poems, lyrics, descriptions) and audio in a shared embedding space — given a poem, find the matching song; given a song, find the matching text.

## How it works

Two frozen pretrained encoders extract features from each modality:

- **Text:** `all-MiniLM-L6-v2` (384-dim)
- **Audio:** `MIT/ast-finetuned-audioset-10-10-0.4593` (AST, 768-dim)

Small trainable projection heads map both into a shared 256-dimensional L2-normalized space. Training minimizes symmetric **InfoNCE loss** over in-batch negatives, with a learnable temperature scalar.

## Getting started

**Local dev (CPU-only Linux)**

```bash
uv sync --extra cpu
```

**Google Colab (CUDA 12.8)**

```python
!pip install uv -q
!uv sync --extra cu128
```

**Train**

```bash
# Cache mel spectrograms from your audio directory
uv run python main.py cache-spectrograms data/audio data/cache

# Train
uv run python main.py train configs/default.toml
```

You'll need a CSV manifest at `data/manifest.csv` with `audio_path` and `text` columns.

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Shared embedding dim | 256 |
| Batch size | 64 |
| Learning rate | 3e-4 (AdamW) |
| Temperature | 0.07 (learnable) |
| Mel bins | 128 |
| Max audio length | 10s |

## Data

A few thousand well-curated pairs is enough. The frozen encoders provide strong representations; the projection heads just learn the alignment. Some options:

- **Free Music Archive (FMA):** creative-commons music with genre/metadata
- **Lyrics datasets:** pair songs with their lyrics
- **LLM-generated descriptions:** write rich text descriptions of audio clips

## Future direction

Once trained, the model is suitable for LLM tool use:

- `search_by_text(query)`: find audio matching a text description
- `search_by_audio(path)`: find text describing a piece of audio
- `similarity(text, audio_path)`: score a text-audio pair
