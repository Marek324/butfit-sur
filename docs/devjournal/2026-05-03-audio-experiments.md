# Audio feature experiments - 2026-05-03

Three feature ablations + one feature-extension experiment. Compared on clean
dev (where the baseline scores AUC=0.972, EER=4.17%) and stress dev (baseline
AUC=0.529, EER=46.67%).

## Variants

| variant | config | feature_dim | rationale |
|---|---|---|---|
| baseline | 5 prosodic + 5 MFCC means | 10 | starting point |
| prosodic_only | 5 prosodic, 0 MFCC | 5 | are prosodics alone enough? |
| mfcc_only | 0 prosodic + 13 MFCC means | 13 | the orthodox MFCC-only approach |
| speaker_features | 5 prosodic + 6 formant/centroid + 5 MFCC | 16 | (running) - speaker-discriminative add-ons |

## Clean dev results (all but speaker_features)

| variant | AUC | EER | TP @ logit=0 | FP @ logit=0 |
|---|---|---|---|---|
| **baseline** | **0.972** | **4.17%** | **8/10** | 2/60 |
| prosodic_only | 0.895 | 19.17% | 1/10 | 5/60 |
| mfcc_only | 0.848 | 20.83% | 5/10 | 1/60 |

Baseline beats both ablations on clean dev. Reading:
- **prosodic_only is significantly worse** - the 5 prosodic features alone (log F0 mean/std, jitter, shimmer, voicing ratio) miss the spectral envelope information MFCCs capture. AUC drops from 0.972 → 0.895.
- **mfcc_only is even worse** - surprisingly. 13 MFCC means without the F0 anchor lose too much. The CMVN per-utterance normalization removes the absolute spectral level (which IS speaker-related), and the means are aggregates so they don't capture short-time variability.
- The combo (5+5) is the right factoring: prosodics anchor the F0/voice quality; MFCC means add spectral envelope. Each carries information the other lacks.

## Stress dev results - extended sweep

After the 2-stage cache made variants cheap, ran the full sweep on stress dev:

| variant | clean AUC | clean EER | stress AUC | stress EER |
|---|---|---|---|---|
| **more_mfcc** (5 prosodic + 13 MFCC) | **0.983** | **3.33%** | 0.521 | 47.08% |
| **speaker_features** (16-d) | 0.985 | 9.17% | 0.529 | 46.67% |
| baseline | 0.972 | 4.17% | 0.529 | 46.67% |
| strong_reg (C=0.1) | 0.968 | 6.67% | 0.529 | 46.67% |
| svm_rbf | 0.933 | 7.50% | **0.537** | **46.25%** |
| prosodic_only | 0.895 | 19.17% | 0.533 | 46.67% |
| mfcc_only | 0.848 | 20.83% | 0.529 | 46.67% |

**Stress dev is uninformative for audio.** All seven variants land in the
0.52–0.54 AUC band - within the standard error for 70 dev samples × 5
augmentations = 350 stress samples. SVM-RBF is nominally the best at stress
AUC=0.537 but EER is still 46% (chance). Audio is fundamentally bottlenecked
by the disjoint augmentation envelope.

**Clean dev DOES separate the variants:**
- `more_mfcc` is the new leader on clean dev (AUC 0.983, EER 3.33%, only 3 errors at logit=0). Adding 8 more MFCC means on top of the 5 prosodic features captures more spectral envelope information.
- `speaker_features` (formants + ratios + centroid) has the highest AUC (0.985) but worst EER@0 in the top tier. Its score distribution shifted; the EER threshold is at -0.71, so logit=0 doesn't sit at the optimal operating point.
- `strong_reg` (C=0.1) is slightly worse than baseline - unlike for image, stronger regularization didn't help here.
- `svm_rbf` is worse on clean (0.933) - the 10-d hand-crafted feature space is already roughly linear-separable, the kernel adds variance without benefit.

## VAD experiments

Two VAD variants, both with their own pre-cache (vad_cfg is part of the pre
hash so they need fresh extraction):

| variant | clean AUC | clean EER | stress AUC | stress EER | confusion @ logit=0 (clean) |
|---|---|---|---|---|---|
| baseline (`energy_zcr`) | 0.972 | 4.17% | 0.521 | 47.08% | TP=8 FP=2 FN=2 TN=58 |
| `vad_strict` (energy_margin=12 dB, zcr_low=0.05, min_speech=200 ms) | 0.967 | 10.00% | 0.500 | 50.00% | TP=8 FP=2 FN=2 TN=58 |
| `vad_off` (no VAD - all frames count as speech) | **0.627** | **39.17%** | **0.668** | **37.75%** | TP=0 FP=11 FN=10 TN=49 |

### Clean-dev verdict: VAD is essential

Removing VAD tanks clean-dev AUC from 0.972 → 0.627 (almost chance) and
produces **TP=0/10 at logit=0** - the model can't separate targets from
impostors at all. Without VAD, MFCC means are computed over the entire
audio including silence, breath, and boundary booms; voicing_ratio degrades
because all "speech" frames count, even unvoiced rumble. The booming
low-frequency content adds a persistent spectral tilt that varies between
recordings and dominates the MFCC signal. So *yes*, VAD does help against
booming sounds - by a wide margin.

`vad_strict` (12 dB energy margin, ZCR floor 0.05, 200 ms min speech)
slightly *underperforms* baseline on clean dev - pulling the floor up by 4 dB
and demanding longer speech runs drops some real speech. The default
thresholds (8 dB / 0.02 / 100 ms) are reasonably tuned for our recordings.

### Stress-dev verdict: a regime flip

Surprising finding under the disjoint stress envelope:

- `vad_off` jumps to AUC **0.668** - best of the entire audio sweep, ~14
  points above any VAD-on variant.
- `vad_strict` drops to AUC 0.500 - exact chance.
- The other VAD-on variants (baseline, more_mfcc, etc.) all sit at 0.52.

What likely happens: under stress (heavy gain attenuation + low SNR), the
adaptive energy threshold mis-fires. With the noise floor close to the speech
floor, the relative-margin rule (`noise_db + 8 dB`) excludes frames that
*are* speech but have been quieted by the augmentation. Aggregating all
frames (vad_off) at least keeps the spectral envelope information intact,
even if it's diluted by silence/noise.

This is a **regime-dependent tradeoff**: VAD wins by 35 AUC points on clean,
loses by 15 AUC points on stress. Eval-set behavior is unknown territory.

### Decision: keep VAD on, ship `more_mfcc`

The clean-dev gap (0.972 → 0.627) is much larger than the stress-dev gap in
the other direction (0.668 → 0.521), and stress dev is itself a *guessed*
distribution - we picked the disjoint envelope; eval may not actually match
it. Plus the 50/50-prior decision rule cares about hard decisions at logit=0,
where vad_off's clean dev confusion is TP=0/10 - unusable.

Keep baseline VAD config. Promote `more_mfcc` (5 prosodic + 13 MFCC means)
on top of it.

## Promoted: `more_mfcc`

Audio baseline updated to **5 prosodic + 13 CMVN-MFCC means** (was 5+5).
Same VAD config, same pyin params - shares pre-cache with the original
baseline so the promotion was instant (no extraction). Re-trained:

- Clean dev AUC 0.983 (was 0.972, +1.1pp)
- Clean dev EER 3.33% (was 4.17%, -0.84pp)
- Confusion at logit=0: TP=7 FP=0 FN=3 TN=60 (vs old TP=8 FP=2 FN=2 TN=58 - fewer total errors, perfect precision but lost one target)
- Stress dev unchanged (within noise band)

Updated `configs/audio_baseline.toml` with comment pointing here. Fusion
re-trained on top - clean dev still AUC=1.0 (biased; fusion learns from dev).
Eval scoring under way to refresh `results/audio_logreg.txt` and
`results/fusion_logreg.txt`.

## Extension experiment: `audio_speaker_features` (re-run after cache refactor)

Adds 6 LPC-derived descriptors on top of the baseline:
- F1, F2, F3 medians (log-Hz) - VTL-covariant
- F2/F1, F3/F1 ratios (log-ratio, **VTL-invariant**)
- spectral centroid mean (log-Hz) - pitch-covariant

Initial cancellation reason was extraction speed - LPC formants per frame
projected ~2 hours for the full training set. Two fixes:
1. **2-stage cache**: pyin/MFCC/formants/centroid all live in `AudioPre`,
   keyed independently of `include` / `mfcc_n_means`. The speaker_features
   variant now hits the cache populated by the baseline retrain → aggregation
   only, ~10 ms per sample, ~1 second across all 1112 training samples.
2. **Formant batching**: LPC only on voiced-in-speech frames; root-finding
   via batched companion-matrix `np.linalg.eigvals`. ~4× speedup on the cold
   path (210 ms → 50 ms per 10-second sample).

Result on clean dev:

| variant | feature_dim | AUC | EER | EER threshold |
|---|---|---|---|---|
| **baseline** | 10 | 0.972 | **4.17%** | -0.375 |
| **speaker_features** | 16 | **0.985** | 9.17% | -0.708 |

Speaker_features wins on AUC (the rank-based metric - formants and ratios *do*
add discriminative information on top of prosodics + MFCC means). Loses on
EER because the score distribution has shifted: the EER threshold drifted to
-0.708 from baseline's -0.375. The model is more confident overall and the
zero-crossing happens above the EER point.

For our 50/50-prior decision rule (`hard_threshold_logit = 0`), AUC and the
threshold-0 confusion matrix are what matter - both metrics still favor the
baseline operationally despite speaker_features having a flatter ROC.

## Decision rule for promotion

Promote the variant with the best **stress dev AUC**, *provided* it doesn't
regress more than ~2% on clean dev. Stress is the more honest generalization
proxy; clean dev saturates and isn't diagnostic.

## What we learned

1. **Both prosodics and MFCCs are necessary** in this 10-d-or-so regime - neither alone reaches the combined performance.
2. **Stress dev separates everything**. All three variants tested so far are statistically indistinguishable under stress (AUC ~0.53). The audio system is fundamentally limited by the harsh disjoint envelope I picked, and feature subset choice within the existing space doesn't change that. To improve under stress we need *different* features (formants, hopefully), not more of the same.
3. **MFCC count doesn't help.** 13 MFCC means alone underperformed 5 MFCC means + prosodics. More features is not better when training set is small.
