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

Downloads audio from two CC-licensed HuggingFace datasets and writes `data/manifest.csv`.

- **LP-MusicCaps-MTT:** ~3K tracks with LLM-generated captions (high-quality)
- **MTG-Jamendo:** ~10.5K tracks with programmatic captions from genre/instrument/mood tags (volume)

```bash
# Download both LP-MusicCaps and Jamendo (~13.5K pairs total)
poe run python main.py ingest

# LP-MusicCaps only (smaller, faster)
poe run python main.py ingest --lp-only

# Jamendo only, limit to N samples
poe run python main.py ingest --jamendo-only --max-samples-jamendo 200

# Limit both sources
poe run python main.py ingest --max-samples-lp 500 --max-samples-jamendo 1000
```

**Bring your own data**

Alternatively, supply a CSV manifest at `data/manifest.csv` with `audio_path`, `text`, and (optionally) `source` columns and place audio files at the referenced paths.

**Train**

```bash
# Cache mel spectrograms from your audio directory
poe run python main.py cache-spectrograms

# Train (saves best.pt to checkpoints/)
poe run python main.py train

# Save checkpoints to a custom directory (e.g. Google Drive)
poe run python main.py train --checkpoint-dir /content/drive/MyDrive/maestroetry-checkpoints

# Resume training from a checkpoint
poe run python main.py train --resume checkpoints/best.pt

# Resume with only model weights (fresh optimizer/scheduler, for changed hyperparameters)
poe run python main.py train --resume checkpoints/best.pt --weights-only
```

**Fine-tuning the audio encoder:** Set `unfreeze_audio_layers` to unfreeze the top N AST transformer layers. These train at `encoder_learning_rate` (default 1e-5) while projection heads use the main `learning_rate`. Gradient clipping is controlled by `max_grad_norm` (default 1.0). This increases VRAM usage; reduce `batch_size` if needed.

Training saves a single `best.pt` checkpoint, overwritten whenever Recall@1 improves. Each checkpoint includes trainable parameters (projection heads and temperature), optimizer state, scheduler state, and the current epoch, so training can be resumed exactly where it left off. Frozen encoder weights are excluded, keeping checkpoints small (~10-15 MB). Use `--weights-only` when resuming with changed hyperparameters (e.g. learning rate, batch size) to load the learned projection weights with a fresh optimizer and scheduler.

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Shared embedding dim | 256 |
| Projection depth | 3 |
| Projection dropout | 0.1 |
| Batch size | 100 |
| Gradient accumulation steps | 4 (effective batch 400) |
| Learning rate | 5e-4 (AdamW) |
| Temperature | 0.07 (learnable) |
| Unfrozen audio layers | 0 |
| Encoder learning rate | 1e-5 |
| Max gradient norm | 1.0 |
| Mel bins | 128 |
| Max audio length | 10s |

## Data

A few thousand well-curated pairs is enough. The frozen encoders provide strong representations; the projection heads just learn the alignment. The built-in ingest pipeline uses two tiers:

- **LP-MusicCaps-MTT:** ~3K tracks with LLM-generated captions (`mulab-mir/lp-music-caps-magnatagatune-3k`)
- **MTG-Jamendo:** ~10.5K tracks with programmatic captions from genre/instrument/mood tags (`vtsouval/mtg_jamendo_autotagging`)

Both are CC-licensed and fetched via the HuggingFace `datasets` library (no YouTube/yt-dlp dependency).

## Attribution

### LP-MusicCaps-MTT

Audio and captions from [LP-MusicCaps: LLM-Based Pseudo Music Captioning](https://arxiv.org/abs/2307.16372) (ISMIR 2023), by SeungHeon Doh, Keunwoo Choi, Jongpil Lee, and Juhan Nam. Licensed under **CC BY 4.0**.

```bibtex
@article{doh2023lp,
  title={LP-MusicCaps: LLM-Based Pseudo Music Captioning},
  author={Doh, SeungHeon and Choi, Keunwoo and Lee, Jongpil and Nam, Juhan},
  journal={arXiv preprint arXiv:2307.16372},
  year={2023}
}
```

### MTG-Jamendo

Audio and tags from [The MTG-Jamendo Dataset for Automatic Music Tagging](https://mtg.github.io/mtg-jamendo-dataset/) (ICML 2019 ML4MD Workshop), by Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra (Music Technology Group, Universitat Pompeu Fabra). Metadata is licensed under **CC BY-NC-SA 4.0**; audio files carry individual CC licenses. Usage is limited to non-commercial research and academic purposes.

```bibtex
@conference{bogdanov2019mtg,
  author    = {Bogdanov, Dmitry and Won, Minz and Tovstogan, Philip and Porter, Alastair and Serra, Xavier},
  title     = {The MTG-Jamendo Dataset for Automatic Music Tagging},
  booktitle = {Machine Learning for Music Discovery Workshop, ICML 2019},
  year      = {2019},
  address   = {Long Beach, CA, United States}
}
```

## Future direction

Once trained, the model is suitable for LLM tool use:

- `search_by_text(query)`: find audio matching a text description
- `search_by_audio(path)`: find text describing a piece of audio
- `similarity(text, audio_path)`: score a text-audio pair
