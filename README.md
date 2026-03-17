# Maestroetry

Contrastive text-audio retrieval model inspired by CLIP. Aligns text (poems, lyrics, descriptions) and audio in a shared embedding space — given a poem, find the matching song; given a song, find the matching text.

## How it works

Two frozen pretrained encoders extract features from each modality:

- **Text:** `all-MiniLM-L6-v2` (384-dim)
- **Audio:** `MIT/ast-finetuned-audioset-10-10-0.4593` (AST, 768-dim)

Small trainable projection heads map both into a shared 256-dimensional L2-normalized space. Training minimizes symmetric **InfoNCE loss** over in-batch negatives, with a learnable temperature scalar.

## Getting started

Install [poe the poet](https://poethepoet.natn.io/) if you haven't already:

```bash
pip install poethepoet
```

**Local dev**

```bash
poe sync-cpu            # CPU-only
poe sync-gpu            # GPU (CUDA 12.8)
poe sync-cpu-ingest     # CPU + ingest support
poe sync-gpu-ingest     # GPU + ingest support
```

**Google Colab (CUDA 12.8)**

```bash
!pip install uv poethepoet -q
!poe sync-gpu-ingest
```

**Ingest data (recommended)**

Downloads audio from two CC-licensed sources and writes `data/manifest.csv`. Requires `ffmpeg` on your PATH.

- **Song Describer Dataset (SDD):** ~1,000 FMA tracks with human-written captions (high-quality anchor)
- **Free Music Archive (FMA):** 8K+ tracks with programmatic captions derived from metadata (volume)

```bash
# Download both SDD and FMA small
poe run python main.py ingest

# SDD only (smaller, faster)
poe run python main.py ingest --sdd-only

# FMA only, limit to N samples
poe run python main.py ingest --fma-only --max-samples-fma 200

# Use the larger FMA medium subset (~25K tracks)
poe run python main.py ingest --fma-subset medium
```

**Bring your own data**

Alternatively, supply a CSV manifest at `data/manifest.csv` with `audio_path`, `text`, and (optionally) `source` columns and place audio files at the referenced paths.

**Train**

```bash
# Cache mel spectrograms from your audio directory
poe run python main.py cache-spectrograms

# Train
poe run python main.py train
```

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

A few thousand well-curated pairs is enough. The frozen encoders provide strong representations; the projection heads just learn the alignment. The built-in ingest pipeline uses two tiers:

- **Song Describer Dataset (SDD):** ~1,000 human-written captions mapped to FMA track IDs (quality anchor)
- **FMA small/medium:** 8K-25K CC-licensed tracks with programmatic captions from genre metadata (volume)

Both are CC-licensed and downloaded directly (no YouTube/yt-dlp dependency).

## Future direction

Once trained, the model is suitable for LLM tool use:

- `search_by_text(query)`: find audio matching a text description
- `search_by_audio(path)`: find text describing a piece of audio
- `similarity(text, audio_path)`: score a text-audio pair
