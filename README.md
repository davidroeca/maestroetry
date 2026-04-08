# Maestroetry

Contrastive text-audio retrieval model inspired by CLIP. Aligns text (poems, lyrics, descriptions) and audio in a shared embedding space — given a poem, find the matching song; given a song, find the matching text.

## How it works

Two pretrained encoders extract features from each modality:

- **Text:** `all-MiniLM-L6-v2` (384-dim, frozen)
- **Audio:** `MIT/ast-finetuned-audioset-10-10-0.4593` (AST, 768-dim, top 2 layers fine-tuned)

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
# Download both LP-MusicCaps and Jamendo (~22-25K pairs total)
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
| Learning rate | 3e-4 (AdamW) |
| LR schedule | Cosine warmup + decay |
| Warmup steps | 200 |
| Epochs | 300 |
| Temperature | 0.07 (learnable) |
| Unfrozen audio layers | 2 |
| Encoder learning rate | 1e-5 |
| Max gradient norm | 1.0 |
| SpecAugment | Enabled (2 freq masks, 2 time masks) |
| Mel bins | 128 |
| Max audio length | 10s |

See [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md) for the full experiment history and results.

## Data

The built-in ingest pipeline uses two tiers:

- **LP-MusicCaps-MTT:** ~3K tracks with 4 LLM-generated caption variants each (`mulab-mir/lp-music-caps-magnatagatune-3k`). All caption variants are used by default, yielding ~12K+ text-audio pairs.
- **MTG-Jamendo:** ~10.5K tracks with programmatic captions from genre/instrument/mood tags (`vtsouval/mtg_jamendo_autotagging`)

Total: ~22-25K training pairs. Both are CC-licensed and fetched via the HuggingFace `datasets` library (no YouTube/yt-dlp dependency).

## Attribution

### [CLAP Model](https://github.com/LAION-AI/CLAP)

```bibtex
@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}
@inproceedings{htsatke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2022}
}
```

## Future direction

Once trained, the model is suitable for LLM tool use:

- `search_by_text(query)`: find audio matching a text description
- `search_by_audio(path)`: find text describing a piece of audio
- `similarity(text, audio_path)`: score a text-audio pair
