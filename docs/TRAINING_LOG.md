# Training Log

Contrastive text-audio retrieval model (symmetric InfoNCE).
Encoders: MiniLM-L6-v2 (text), AST-finetuned-audioset (audio).
Dataset: LP-MusicCaps + Jamendo.

## Summary

| Run | Change | Best Epoch | t2a R@1 | a2t R@1 | t2a R@5 |
|-----|--------|-----------|---------|---------|---------|
| 1 | Dropout only (depth=2) | 5 | 0.0003 | 0.0000 | 0.0010 |
| 2 | Deeper heads (depth=3) | 100 | 0.2178 | 0.2229 | 0.3403 |
| 3 | Unfreeze 2 AST layers | 100 | 0.2645 | 0.2673 | 0.3674 |
| 3-ext | Extended to 200 epochs | 200 | 0.2753 | 0.2805 | 0.3649 |
| 4 | Unfreeze 4 AST layers | 80 | 0.2709 | 0.2773 | 0.3610 |
| 5 | Cosine LR schedule | 165 | 0.2894 | 0.2837 | 0.3658 |
| 6 | SpecAugment | 265 | 0.2670 | 0.2657 | 0.3569 |
| 7 | Multi-caption dataset | 270 | 0.9949 | 0.9831 | 1.0000 |

## Run 1: Dropout only

- **Config:** `projection_depth=2`, `dropout=0.1`, frozen encoders, `lr=5e-4`
- **Epochs:** 100 (best at 5)
- **Goal:** Isolate the effect of dropout regularization on a shallow projection head.
- **Result:** R@1 at final epoch not saved due to software bug fixed by run 2 (but was lower than run 2).

## Run 2: Deeper projection heads

- **Config:** `projection_depth=3`, `dropout=0.1`, frozen encoders, `lr=5e-4`
- **Epochs:** 100
- **Goal:** Establish whether a deeper projection head improves alignment.
- **Result:** First real learning saved. Depth=3 became the standard going forward.

## Run 3: Unfreeze top 2 AST layers

- **Config:** `projection_depth=3`, `unfreeze_audio_layers=2`, `lr=3e-4`, `encoder_lr=1e-5`
- **Epochs:** 100 (later extended to 200 as run3-extended)
- **Goal:** Allow the audio encoder to adapt its top layers with a differential learning rate.
- **Result:** Modest improvement. Extended training helped slightly (0.2645 to 0.2753).

## Run 4: Unfreeze top 4 AST layers

- **Config:** `unfreeze_audio_layers=4`, otherwise same as run 3
- **Epochs:** 100 (best at 80)
- **Goal:** Test whether unfreezing more layers helps.
- **Result:** No improvement over run 3. More unfreezing did not help; stuck with 2.

## Run 5: Cosine LR schedule

- **Config:** Same as run 3 plus `lr_schedule="cosine"`, 300 epochs
- **Resume:** From run 3-extended best checkpoint (weights only)
- **Goal:** Isolate the gain from proper LR decay vs. constant post-warmup.
- **Result:** Small improvement to 0.2894, best of the single-caption runs.

## Run 6: SpecAugment

- **Config:** Run 5 settings plus `spec_augment=true` (freq/time masking)
- **Resume:** From run 5 best checkpoint (weights only)
- **Goal:** Add data augmentation to improve generalization.
- **Result:** Slight regression to 0.2670. SpecAugment alone did not help at this data scale.

## Run 7: Multi-caption dataset

- **Config:** Run 6 settings plus `warmup_steps=200`
- **Dataset change:** Re-ingested LP-MusicCaps with all caption variants (all 4 captions per track, ~22-25K pairs vs. ~13.5K)
- **Resume:** From run 6 best checkpoint (weights only)
- **Goal:** Expand training signal by using all available captions per audio track.
- **Result:** This may be an over-fitting result.

## Key Takeaways

1. Architectural and schedule changes (runs 3-6) yielded diminishing returns in the 0.26-0.29 range.
