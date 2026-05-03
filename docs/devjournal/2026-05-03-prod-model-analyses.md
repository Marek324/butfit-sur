# Production-model analyses — 2026-05-03

Five analyses against the promoted production stack (audio more_mfcc 18-d,
image HOG 8×8 with C=0.1, fusion zscore-logreg). Outputs in `artifacts/`,
scripts in `scripts/`.

## 1. Coefficient inspection (`coef_inspect.py`)

### Audio (18-d, intercept = -2.18)

The model's strongest weights — sorted from positive (favors target) to
negative (favors impostor) — say something coherent about m431:

| feature | coefficient |
|---|---|
| voicing_ratio | most positive (favors target) |
| jitter_local | strongly positive |
| mfcc_mean_0 (DC / overall energy after CMVN) | positive |
| mfcc_mean_3, mfcc_mean_4 | small positive |
| mfcc_mean_2, mfcc_mean_5–9 | mostly small |
| mfcc_mean_10 | -0.62 |
| **shimmer_local** | **-1.43** |
| **log_f0_mean** | **-2.06** |
| **log_f0_std** | **-2.78** (most negative — favors impostor) |

The big negative weights on `log_f0_mean` and `log_f0_std` are the most
telling: the impostor pool has *higher and more variable* F0 than m431, so
lower / steadier F0 → higher target score. This matches what we'd expect
from a single male target voice mixed against a diverse impostor pool with
more females (higher F0). `shimmer_local` being negative says m431 has
*more* amplitude perturbation than the impostor average — more vocal-fry-like
or rougher.

### Image (HOG 2916-d, intercept = -6.01)

Weights are sparse-ish: median |w| = 0.049, max |w| = 0.272. Top positive
HOG dims cluster in the **1700–2200** range (feature_vector layout = blocks
× cells/block × orientations; 1700–2200 is roughly the lower-right face
region — chin / mouth area for a face-shaped 80×80 input). Top negatives
are scattered.

The intercept of -6.01 is striking: pre-bias, the model strongly defaults
to "non-target", and only when the HOG features push the linear combination
above 6 does it predict m431. That's the model leaning hard on
`class_weight='balanced'` to recalibrate the threshold to logit=0.

### Fusion (zscored inputs, intercept = -3.41)

| input | weight |
|---|---|
| audio_score (zscored) | +1.65 |
| **image_score (zscored)** | **+2.21** |

Fusion leans on **image** by 1.34×. The zscore normalization equalizes
input magnitudes, so the weight ratio is the actual relative trust. This
matches the eval-set Pearson(image, fusion) = 0.85 vs Pearson(audio,
fusion) = 0.76 from the score-correlation analysis: when audio and image
disagree, fusion sides with image more often.

## 2. Per-segment disagreement on eval (`eval_disagreement.py`)

Hard-decision combinations on the 1346 eval segments:

| audio | image | fusion | count | % | label |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 1049 | 77.9% | NNN |
| 0 | 0 | 1 | 16 | 1.2% | NNT |
| 0 | 1 | 0 | 29 | 2.2% | NTN |
| 0 | 1 | 1 | 21 | 1.6% | NTT |
| 1 | 0 | 0 | 96 | 7.1% | TNN |
| 1 | 0 | 1 | 58 | 4.3% | TNT |
| 1 | 1 | 0 | 0 | 0.0% | TTN |
| 1 | 1 | 1 | 77 | 5.7% | TTT |

Reading:
- **TTN = 0**: when audio AND image both say target, fusion *never* overrides.
- **NNT = 16**: 16 eval segments where both unimodals say non-target but fusion flips them to target. Both per-modality scores were close to zero and the fusion linear combination pulled them across.
- **TTT = 77**: the high-confidence target segments — both modalities agree, fusion confirms. These are the safest target predictions in our submission.
- **Disagreement (204 segments / 15.2%)**:
  - Audio says T, image says N: 154 cases. Fusion sides with image (96, 62%) more than with audio (58, 38%).
  - Audio says N, image says T: 50 cases. Fusion sides with image (21, 42%) less than with audio (29, 58%).
  - Net: fusion goes with the image direction in 117 disagreement cases vs 87 with audio — consistent with the +2.21 vs +1.65 weight pattern.

## 3. CV-OOF fusion (`cv_oof_fusion.py`)

5-fold stratified CV (one m431 session in dev → session-grouped CV would
produce single-class folds, so stratified is the right choice). Per-modality
models are unchanged (already trained on `train` only); only the fusion
linear combination is re-fit per fold.

Per-fold fusion weights are stable:

| fold | audio_w | image_w | intercept |
|---|---|---|---|
| 0 | 1.68 | 1.95 | -3.20 |
| 1 | 1.46 | 2.10 | -3.16 |
| 2 | 1.53 | 2.11 | -3.18 |
| 3 | 1.54 | 2.10 | -3.13 |
| 4 | 1.58 | 2.01 | -3.15 |

Out-of-fold metrics (unbiased dev estimate):
- **AUC = 1.0000**
- **EER = 0.00%**
- Confusion @ logit=0: TP=10, FP=0, FN=0, TN=60

**Identical to the production (biased) fusion metrics.** Fusion isn't
overfitting dev — the dev set is genuinely fusion-separable. Per-modality
errors complement each other (see analysis 4 below), so any reasonable
weighting recovers all 70 dev samples correctly.

So the fusion's "biased optimistic" caveat is technically true (training
data leaked into evaluation), but for this dev split the bias is zero —
the held-out estimate matches.

## 4. Dev error correlation (`dev_error_correlation.py`)

At logit=0 on dev:
- Audio: 67/70 correct (95.7%) — TP=7, FP=0, FN=3, TN=60
- Image: 69/70 correct (98.6%) — TP=10, FP=1, FN=0, TN=59

2×2 contingency:

|              | image ✓ | image ✗ |
|---           |---:|---:|
| **audio ✓** | 66 | 1  |
| **audio ✗** | 3  | 0  |

- Both correct: 66 (94.3%)
- Audio rescues image: 1 (the one image FP — audio correctly says non-target)
- Image rescues audio: 3 (the three audio FNs — image correctly catches the targets)
- Both wrong: **0**

**Phi correlation of correctness: -0.025** — essentially independent
errors, very mildly complementary. Fusion-rescuable mistakes: 4. Unrescuable
mistakes: 0. This is exactly the regime where score-level fusion shines: at
least one modality has the answer for every error case.

This is *why* fusion gets AUC=1.0 / EER=0% on dev (and via CV-OOF — see
above). The two modalities pick up genuinely different identity cues
(eval-set audio↔image Pearson = 0.32) and their errors don't coincide on
this dev split. Whether this complementarity holds on the eval set is the
question for May 5th's answer key.

## 5. Score calibration on dev (`calibration_dev.py`)

Reliability diagrams for `sigmoid(logit)` from each modality, binned into
10 equal-width bins on [0, 1]. Audio + image use the production models
trained on `train` only (honest dev scores). Fusion uses CV-OOF logits
(unbiased) — same 5-fold stratified split as analysis 3.

| modality | Brier | ECE |
|---|---:|---:|
| audio | 0.0455 | 0.1263 |
| image | 0.0249 | 0.0858 |
| fusion (OOF) | 0.0121 | 0.0587 |

Brier and ECE both rank **fusion < image < audio**, matching the AUC
ranking. Fusion is the best-calibrated by a clear margin.

Reading the diagrams:

- **Audio** is messy in the middle. Non-targets pile cleanly at low P
  (37+13+6 = 56 in the lowest three bins, all observed freq 0). A bump at
  P≈0.36 (5 samples, observed freq 0.6 — three of those five are actually
  targets the model is unsure about), a dip at P≈0.45 (2 non-targets), then
  the high bins (P≥0.65) are all observed freq 1.0. Audio under-extends
  itself — the targets it gets right show up across P=0.7 to P=0.92, but
  observed freq saturates at 1.0 in every one of those bins. Net pattern:
  **slightly under-confident** on its hardest 5 cases (the P≈0.36 bin) and
  **over-confident in the high range** (predicted 0.7 means observed 1.0).
- **Image** is very binary. P<0.45 → all non-target (n=58). P>0.6 → all
  target (n=8) except a single n=3 transition bin at P≈0.55 with observed
  freq 0.67. The model essentially refuses to give intermediate
  probabilities; mid-range bins are nearly empty. ECE is moderate not
  because the model is poorly calibrated, but because in the high range it
  says "P≈0.85" while observation is 1.0.
- **Fusion (OOF)** is the cleanest. 48 non-targets piled at P≈0.02; 8
  more in the low bins. Targets pack at P≈0.85–0.95 with observed freq
  1.0. Same over-confidence-in-the-high-range issue (predicting 0.9
  instead of 1.0) — but the resulting ECE is the lowest of the three.

The systematic "predicted 0.85 → observed 1.0" pattern across all three
modalities is the LR side-effect of `class_weight='balanced'`: the loss
is rebalanced so the sigmoid output is the posterior under a synthetic
50/50 prior, not the natural 10/60 dev frequency. The decision threshold
(logit=0 ↔ P=0.5) is what we actually care about for the submission's
hard-decision column, and that threshold is correct under the 50/50 prior
the assignment specifies. So this calibration story is a documentation
note, not a fix-it.

## Practical takeaways for the documentation

- **Audio's "speaker-defining" features are clearly speaker-vs-pool, not in absolute units.** The model says "low and steady F0 → m431"; this is "compared to the impostor pool", not in any universal sense. If the eval impostor distribution has more low-F0 male voices, audio confidence will drop on those.
- **Fusion leans image-heavy (1.34×).** Defensible: image is more accurate on dev (98.6% vs 95.7%) and has a wider score range, so it carries more useful signal per unit of zscored input. The fusion weight ratio reflects this honestly.
- **The dev-fusion bias caveat doesn't bite here** — CV-OOF gives identical metrics. Worth keeping the caveat in the doc anyway because eval may behave differently.
- **Dev errors are essentially independent** (φ ≈ 0). Fusion's job on dev is to combine strong-but-imperfect classifiers; on harder eval data the real test is whether they continue to be wrong on different samples.
- **Sigmoid output is over-confident in the high range across all three modalities** — Brier 0.045 / 0.025 / 0.012 (audio / image / fusion). The 50/50 hard-decision threshold (logit=0) is unaffected; the calibration deviation is a side-effect of `class_weight='balanced'` rebalancing the LR loss. Worth a one-line caveat in the doc rather than a calibration head.
