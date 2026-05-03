# Augmentation strategy — 2026-05-03

## Offline (write to disk) instead of online (apply at training time)

Originally augmentation was inside the trainer — pull each sample, apply random transforms, extract features, repeat. Switched to: pre-augment to disk via `scripts/augment_dataset.py`, training enumerates the augmented files like any other sample.

Why:
- **Caching covers augmentation.** The per-sample feature cache keys on filename stem; augmented files have unique stems (`<orig>_aug{kk}`) so they cache normally. Online aug would re-extract every run.
- **Augmented data is inspectable.** Can sanity-check noise level by listening to the WAV / opening the PNG.
- **Reproducibility.** Once `dataset/*_aug` is populated, training is fully deterministic from those bytes.
- **Decouples experiments.** Iterating on classifier choice while holding the augmented dataset constant is straightforward.

Cost: ~660 MB extra disk for audio + image augmented data. Acceptable; data is gitignored anyway.

The cache is mtime-aware: re-running `augment_dataset.py` with a different envelope auto-invalidates stale features. Discovered this was needed mid-iteration when re-augmenting silently used cached old features → "fixed" results that weren't actually re-trained.

## Envelope choice — what's in and what's deliberately out

**Audio (`configs/augment_audio.toml`):**
- Time-stretch [0.9, 1.1] (pitch-preserving, so prosodic features unchanged in expectation)
- Gain ±6 dB, polarity flip, random crop ≥5 s, peaking EQ
- Additive white Gaussian noise at SNR ∈ [15, 30] dB (added after seeing the assignment promise of "noisy eval data")

**Deliberately omitted:**
- Pitch shift — would change log-F0 mean/std, which IS a speaker-defining feature
- Heavy reverb — biases jitter/shimmer estimates
- SpecAugment — masks parts of the spectrogram, also degrades CMVN-MFCC means
- Pink/colored noise — overkill for the SNR generalization concern

**Image (`configs/augment_image.toml`):**
- Horizontal flip (faces are bilaterally near-symmetric)
- ±10° rotation, ±5% translation
- Brightness 0.85–1.15, contrast 0.85–1.15
- Per-pixel-per-channel Gaussian noise σ ∈ [0, 0.04]

**Deliberately omitted:**
- Vertical flip — faces aren't vertically symmetric
- Large rotations — head doesn't naturally tilt 90°
- Hue rotation — would change skin tone, which is identity

The omissions matter as much as the inclusions for documentation: pitch shift is the textbook "I added augmentation and it ruined my speaker model" mistake.

## The asymmetric-noise incident

After adding noise to the audio envelope, dev metrics dropped:
- AUC: 0.972 → 0.888
- TP @ logit=0: 8/10 → **4/10**
- EER threshold drifted to -1.355 (way off zero)

Initially looked like a 50/50-prior bug. Wasn't — `class_weight='balanced'` was correctly enforced. The actual cause was hidden in the augment config: `n_per_nontarget = 0` (set early when noise wasn't yet in the envelope, and class imbalance was the only concern). After noise was added, the train distribution looked like:

| | clean | noisy | total | noisy fraction |
|---|---|---|---|---|
| target     |  20 | 300 |  320 | **94%** |
| non-target | 132 |   0 |  132 | **0%**  |

The classifier picked up "is this noisy?" as a partial discriminator with noisy ⇒ target. At test time on clean dev, that cue is absent → many targets fell below the decision threshold.

Fix: set `n_per_nontarget = 5`, which gives both classes ≥83% noisy fraction. Re-trained:

- AUC: 0.888 → **0.972**
- EER: 20% → **4.17%**
- TP @ logit=0: 4/10 → **8/10**

## Lesson

When augmentation is class-asymmetric, the model can pick up the asymmetry as a discriminator even when classes are perfectly weight-balanced by `class_weight`. Always check: *what fraction of each class has each augmentation applied?* — aim for parity. The failure mode looks like a calibration / prior issue but isn't.
