# 2-stage audio cache + experiments package - 2026-05-03

## The audio extraction bottleneck wasn't formants

Original suspicion: LPC-per-frame for formant extraction was slow. Profiling
on a 7-second m431 sample said otherwise:

```
VAD                : 0.016 s
pyin               : 4.122 s   ← bottleneck
MFCC               : 0.047 s
formants per_frame : 0.212 s   ← actually fine
spectral_centroid  : 0.032 s
```

**pyin dominates** - 95% of the per-sample wall time is `librosa.pyin`. The
formant work I worried about is 20× cheaper than pyin.

Re-running pyin every time we change `include` or `mfcc_n_means` was the
real waste. Each new audio variant burned ~40 minutes redoing pyin on the
same waveforms.

## 2-stage per-sample cache

`audio_features.py` now exposes:

```
extract_pre(x, sr, vad_cfg, feat_cfg) -> AudioPre   # heavy: pyin + MFCC + formants + centroid
aggregate(pre, feat_cfg)              -> ndarray    # cheap: select / mean / median
extract(x, sr, vad_cfg, feat_cfg)     -> ndarray    # convenience wrapper that runs both
```

`AudioPre` is a frozen dataclass holding all per-frame intermediates:
`sp_mask, f0, voiced_flag, voiced_in_speech, rms, mfcc, formants_per_frame,
centroid_per_frame`. Each entry is just a numpy array, serializable as a
single `.npz`.

In `experiments/extraction.py` the cache became 2-stage:

```
artifacts/cache/audio/<vector_cfg_hash>/<stem>.npy   ← final feature vector
artifacts/cache/audio_pre/<pre_cfg_hash>/<stem>.npz  ← AudioPre intermediates
```

`pre_cfg_hash` is keyed only on what affects the *contents* of `AudioPre`:
`vad_cfg`, pyin params (`f0_method`, `f0_min_hz`, `f0_max_hz`), MFCC dim
(`mfcc_n`, `mel_filters`, `cmvn`). Crucially **NOT** `include` or
`mfcc_n_means` - those only affect the cheap aggregation step.

Lookup order in `_audio_two_stage_cached`:
1. Vector cache hit → return final array.
2. Pre cache hit → load AudioPre, run aggregate, save vector. ~10 ms.
3. Both miss → load WAV, run pyin/MFCC/formants/centroid, save pre, aggregate, save vector. ~2 s.

Both caches are mtime-aware: re-augmentation invalidates them automatically.

## Measured speedup

On a real m431 dev sample:

| | full extract (pre+aggregate) | aggregate only (pre cached) |
|---|---|---|
| baseline 10-d | 2.6 s | **0.1 ms** |
| speaker_features 16-d | 2.0 s | **0.3 ms** |

Five-sample test against the dev split:

```
baseline 5 samples cold              : 13.08 s
baseline 5 samples (vector cache)    :  0.001 s
prosodic_only 5 samples (pre HIT)    :  0.013 s   ← was ~50 min projected
```

For a typical "swap include list" experiment over the full augmented training
set (1182 samples), the new cost is ~10–30 seconds instead of ~30–50 minutes.
This is what makes running many feature variants tractable.

## Formant-extraction batching (~4×)

Same module also got a formant-extraction speedup:

- LPC is now computed only on **voiced-in-speech** frames (skips ~50% of frames
  on typical recordings - non-voiced frames stay NaN-padded in the output).
- Polynomial roots are found via batched companion-matrix `np.linalg.eigvals`
  on a 3-D tensor instead of `np.roots` per polynomial.

The combined effect cuts formant extraction from ~210 ms → ~50 ms per 10-second
sample. Less of a win than the pre-cache, but free, and makes future feature
sweeps that *do* change pyin params still tolerably fast.

## Experiments package split

Old `src/experiments.py` was 413 lines and growing. Split into a package:

```
src/experiments/
  __init__.py    ← re-exports the public API
  extraction.py  ← extract_set, the 2-stage cache, feature_config_hash
  runners.py     ← run_audio_or_image, run_fusion, run_experiment, score_unlabeled
  compare.py     ← compare_runs (sidecar table)
```

All callers (scripts/train.py, evaluate.py, run_experiments.py, plot_curves.py,
plot_experiment_summary.py) only depend on the public names re-exported from
`__init__.py`, so the move was a pure refactor - no test failures, no
behavior change, ruff + ty clean.

## What this enables

- Many audio variants in minutes, not hours. The cancelled
  `audio_speaker_features` extension is now tractable - re-aggregating the
  16-d vector from cached pre takes a few seconds across the whole training
  set.
- VAD experiments (`audio_vad_off`, `audio_vad_strict`) become first-class
  variants - they each need their own pre cache (vad_cfg is a pre param) but
  any subsequent feature changes on top reuse the new cache.
- Future experiments that change classifier-only (no feature change) hit both
  caches and are instant.
