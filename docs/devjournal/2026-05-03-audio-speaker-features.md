# Audio speaker features (formants + spectral centroid) — 2026-05-03

## Why these features

A user-suggested framework: speaker-discriminative features have known relationships to specific augmentations. Each augmentation has a **covariant** feature (changes proportionally) and sometimes a derivable **invariant** (cancels the augmentation factor):

| augmentation | covariant feature | invariant |
|---|---|---|
| pitch shift | F0, log F0 | jitter_local (period ratio) |
| time stretch | speaking rate | voicing_ratio (already in baseline) |
| VTLP / formant shift | F1, F2, F3 | F2/F1, F3/F1 ratios |
| gain | RMS levels | jitter, shimmer (ratios) |
| AWGN | spectral centroid (drifts up) | high-SNR features (when intact) |

**The point:** features that are *covariant* under our augmentations behave predictably. Features that are *invariant* under VTLP/gain/EQ are the ones that should survive the stress envelope.

The current 10-d baseline already covers pitch (log F0), pitch perturbation (jitter), level perturbation (shimmer), and voicing rate. It's missing the **formant axis** entirely — and formants are the single biggest speaker-discriminative DSP feature available without learned models.

## What was added

In `src/audio_features.py`:

```
f1_median, f2_median, f3_median      log-Hz median of LPC formants 1–3 over voiced-in-speech frames
f2_f1_ratio, f3_f1_ratio             log-Hz median of formant ratios — VTL-INVARIANT
spectral_centroid_log_mean           log-Hz mean of spectral centroid over voiced-in-speech frames
```

Implementation: per-frame LPC (order 12, hamming window) → roots → angle → Hz; filter to 90–7900 Hz; sort and take lowest 3. Formant ratios computed per-frame in log space, then median; this is robust to the occasional spurious LPC pole.

Total: 6 new descriptors. Combined with the 5 baseline prosodics + 5 MFCC means → 16-d feature vector.

## Why ratios matter

Under VTLP (vocal tract length perturbation) the formant frequencies scale by a common factor α: F_n' = α·F_n. So:
- `log F_n` shifts additively by `log(α)` (covariant — the model sees a translation)
- `F_n / F_m` is unchanged (invariant)
- `log(F_n / F_m)` is unchanged (linear relationship preserved)

The ratios are therefore *speaker-discriminative without depending on VTLP-style augmentation matching*. Even though we don't apply VTLP today, the same logic applies to natural between-utterance pitch / VTL drift in the eval set.

## Variant config

`configs/experiments/audio_speaker_features.toml` — extends baseline with the 6 new features.

Comparison plan: train on the same augmented dataset, evaluate on stress dev. If the 16-d variant beats the 10-d baseline on stress AUC by a meaningful margin, promote.

## What to expect

Hypothesis: stress dev AUC should improve. The baseline collapses to 0.529 on stress; some of that is from MFCC means shifting under EQ/noise. Formants are computed from spectral envelope shape, which is more robust to additive noise (noise adds to the spectrum but the envelope peaks remain). Ratios add VTL-invariance for free.

Risk: LPC formant estimation is noisy on short frames. Median-over-voiced helps, but per-utterance estimates can still be off by 100–200 Hz. The model has to learn to use noisy features — but with `class_weight='balanced'` and L2 reg, that's the standard small-data regime.
