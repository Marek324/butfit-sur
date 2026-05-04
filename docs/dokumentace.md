---
title: "SUR 2025/2026 Project"
author:
 - "Marek Hric (xhricma00)"
 - "Jakub Komanický (xkomanj00)"
date: "2026-05-04"
geometry: margin=2cm
fontsize: 10pt
---

# 1. Augmentation

The training pool is small (30 base paired stems of *m431* across two
sessions, ~190 non-target stems from disjoint speakers) and the
assignment warns the eval set will contain input-quality variations not
seen at training. Offline augmentation (`scripts/augment_dataset.py`)
addresses both: it enlarges the effective training set and regularises
against acoustic and visual distortions. Audio
augmentation applies time stretch (0.9–1.1×), gain (±6 dB), polarity
flip ($p=0.5$), peaking-band equalisation (EQ; ±3 dB at 200–4000 Hz) and additive White Gaussian Noise (WGN)
(SNR 15–30 dB). image augmentation applies horizontal flip ($p=0.5$),
rotation (±10°), translation (±5 %), brightness and contrast scaling
(0.85–1.15×) and per-pixel WGN ($\sigma\in[0,0.04]$). Each base target
sample receives 15 augmented variants and each non-target receives 5.

# 2. Evaluation Metrics

Four quantities are reported throughout. **AUC** is the primary
rank statistic, threshold-free and robust to class imbalance. Most
variant-selection decisions (audio feature subsets, image
regularisation, GMM-UBM mixture counts) were made on AUC.
**Equal Error Rate (EER) and the EER threshold** summarise the
symmetric-cost operating point under prior 0.5. Drift of the EER
threshold away from zero flags miscalibration, and we rejected the
MAP-adapted GMM-UBM variant on this signal alone (its EER threshold
sat at $+0.25$ across the entire $\tau$ sweep). **Confusion at
$\mathrm{logit} = 0$** (TP/FP/FN/TN) gives the operationally relevant
quantities under the assignment's hard rule. **Precision and F$_1$**
become decisive when stress-dev separates the fusion variants:
`fusion_3way` reaches precision 0.70 and F$_1$ = 0.70 on stress dev
versus 0.31 / 0.47 for the more recall-favouring fusions, which is
what motivates keeping it in the submitted lineup despite a slightly
lower stress AUC than `fusion_hybrid_image`.

# 3. Image System

Image features are 9-bin Histograms of Oriented Gradients over 8×8-pixel cells with 2×2-cell L2-Hys normalised
blocks on 80×80 grayscale crops, producing 2916-d vectors. A balanced
logistic regression with L2 regularisation ($C=0.1$) classifies them.
Hard decisions threshold $\mathrm{decision\_function}$ at zero. The
progression from the syllabus baseline isolates three design changes:

| Step | Description | AUC | EER |
|------|-------------|----:|----:|
| 1 | Baseline: pixels → PCA(D=21) → Gaussian LDA + 2 IRLS | 0.943 | 20.00 % |
| 2 | HOG → PCA(D=21) → Gaussian LDA + 2 IRLS | 0.977 | 9.17 % |
| 3 | HOG (full 2916-d) → balanced LogReg, $C=0.1$, no augmentation | 0.903 | 21.67 % |
| 4 | + offline augmentation (submitted) | 1.000 | 0.00 % |

Step 1→2 isolates the HOG vs. raw-pixel feature gain at fixed
dimensionality. Step 2→3 is a *regression*: dropping PCA in favour of
full-dimensional HOG with L2 leaves the threshold-zero hard rule
mis-calibrated at only 10 target training images. Step 3→4 shows that
augmentation is what makes the high-dimensional architecture viable. Full-dimensional + L2 in isolation is worse than the PCA-bridged
baseline classifier of step 2.

# 4. Audio System

Both audio systems share the same frontend: an energy + zero-crossing-rate
voice activity detector (VAD), 13 mel-frequency cepstral coefficients
at 25 ms / 10 ms framing with per-sample cepstral mean–variance
normalisation, and probabilistic-YIN F$_0$ estimation over voiced
frames in 60–400 Hz. From the F$_0$ track and frame energies
we compute five prosodic statistics ($\log F_0$ mean and standard
deviation, jitter, shimmer, voicing ratio).

**Path A, aggregated features and logistic regression.** The 5
prosodic statistics and the 13 MFCC means form an 18-d sample-level
vector classified by balanced LogReg ($C=1.0$). This is shipped as
`audio_logreg.txt`.

**Path B, Gaussian Mixture Model with universal background, MAP-free.**
Following we fit a 16-mixture diagonal-covariance
universal background model (UBM) on non-target frames and a 4-mixture
diagonal-covariance target GMM on *m431* frames, *independently* by EM.
We chose MAP-free over MAP adaptation because ~32 k augmented target
frames are plentiful at the per-component level ($\alpha_k\to 1$
regardless of relevance factor), and a relevance-factor sweep over
$\tau\in\{5, 10, 20, 100, 500, 2000, 10000\}$ produced an AUC plateau
of 0.98 with the threshold-zero hard rule miscalibrated (EER threshold
drifted to $+0.25$). The per-sample score is the frame-averaged
log-likelihood ratio. We did not attempt i-vector front-end factor
analysis, the standard upgrade over GMM-UBM, because it requires
hundreds of background speakers for the
total-variability matrix to be estimable, whereas our pool has 19
distinct non-target speakers.

**The VAD-empty pathology.** Auditing eval revealed 79 of 1346 samples
(5.9 %) producing identical audio score $\approx +1.535$. These
extract to all-zero feature vectors: under heavy gain attenuation or
short voiced segments, the energy-relative VAD trims every frame,
leaving prosodic and MFCC statistics at their zero-fallback. The
trained classifier maps this zero input to a bias-driven default that
happened to be positive, mislabelling all 79 as target.
Spectral-flatness and voiced-fraction analysis confirmed all 79 are
real speech, not white noise (median voiced fraction 0.60). The
sentinel default was discarding genuine information. The same
pathology contaminates training: 121 of 1112 augmented training
samples also extract to zero vectors, with the target class
disproportionately affected (81 zero-frame target augmentations vs. 40
non-target), so the model also *learned* a small "zero $\Rightarrow$
target" correlation as a class-asymmetric augmentation artefact.

**Hybrid VAD-on / VAD-off fallback.** The fix is a routing scheme
(`scripts/audio_gmm_hybrid.py`): an auxiliary 16+4 GMM-UBM is trained
on the same MFCC frames *without* applying VAD. At scoring time the
submitted VAD-on model is consulted first, and the no-VAD model takes
over only when VAD has trimmed every frame. The hybrid is shipped as
`audio_gmm_hybrid.txt` and short-circuits both the eval-time sentinel
defaults and the train-time spurious correlation at scoring.

The audio progression covers both clean and stress development splits:

| Step | Description | Clean AUC | Stress AUC |
|------|-------------|----------:|-----------:|
| 1 | Baseline: course MFCC + 2 GMMs M=30 (no VAD, no CMVN) | 0.978 | 0.799 |
| 2 | + frame-energy VAD + per-sample CMVN, same 2 GMMs M=30 | 1.000 | 0.730 |
| 3 | Submitted GMM-UBM (MAP-free, sentinel for VAD-empty) | 0.997 | 0.463 |
| 4 | Hybrid VAD-on/off (submitted) | 0.997 | **0.870** |

VAD and CMVN saturate clean dev (steps 1→2) but degrade the stress
split because they introduce the failure mode that step 4 must repair.
The hybrid recovers the baseline's stress robustness without
sacrificing clean-dev performance.

# 5. Multimodal Fusion

Fusion is performed at the score level. For each per-modality system,
dev-set scores are z-normalised. A balanced logistic regression is
fitted on the concatenated z-scores and the ground-truth labels. The
classifier is then applied to z-normalised eval scores. Score-level was
preferred to feature-level because the audio (18-d) and image (2916-d)
spaces are extremely imbalanced: stacking them with ~1100 training
samples invites the curse of dimensionality, whereas a 2- or 3-feature
fusion logreg has too few parameters to overfit the 70-sample dev split.
We submit three variants: `fusion_logreg.txt` over $(z(\text{audio\_logreg}),
z(\text{image}))$. `fusion_hybrid_image.txt` over
$(z(\text{audio\_gmm\_hybrid}), z(\text{image}))$. And `fusion_3way.txt`
over $(z(\text{audio\_logreg}), z(\text{audio\_gmm\_hybrid}),
z(\text{image}))$. The three-way variant tests whether the two audio
paths capture complementary signal, since aggregated 18-d features and
frame-level GMM likelihoods are structurally different objects.

# 6. Validation Strategy

Clean `*_dev` (n=70) saturates for image and fusion, so variant
selection ran on a *stress* dev built by re-augmenting dev with a
*disjoint* envelope (gain $-15$ to $-8$ dB, SNR 3–13 dB, time-stretch
0.78–0.88, image brightness 0.45–0.70, per-pixel noise $\sigma\in[0,0.10]$),
which converts each sample from interpolation to extrapolation under
the trained model. Each variant in the per-modality sweeps was
promoted into the submitted lineup on its stress AUC, with stress EER and
threshold-zero confusion as tiebreakers. The 50/50-prior decision
rule is enforced at training (`class_weight='balanced'`) and
inference (`hard_threshold_logit = 0.0`).

# 7. Results

The submission ships one image system, two audio systems and three
fusion variants, 6 result files in total.

Based on our testing, all three fusion variants saturate clean dev.
The image system is essentially perfect on clean conditions, the
GMM-UBM-hybrid audio system is close behind, and the LogReg audio
system trails by a few AUC points. Stress dev separates the variants
clearly, and the spread there is the better generalisation indicator
than clean-dev AUC. Under stress, the hybrid audio path strongly
outperforms the LogReg audio path, and the fusion variants that use it
inherit that advantage. We expect `fusion_hybrid_image` to maximise
stress AUC and `fusion_3way` to dominate on precision-weighted
metrics: `fusion_3way` produces a substantially more balanced
confusion matrix on stress, trading a small AUC decrement for a large
precision gain. Under an operating point that penalises false
positives more than the assignment's symmetric costs, `fusion_3way`
would be the strongest single submission.

# 8. Reproduction

```bash
uv run scripts/augment_dataset.py --config configs/augment_audio.toml
uv run scripts/augment_dataset.py --config configs/augment_image.toml
uv run scripts/train.py --config configs/audio.toml
uv run scripts/train.py --config configs/image.toml
uv run scripts/train.py --config configs/fusion.toml
uv run scripts/evaluate.py --config configs/audio.toml --split eval
uv run scripts/evaluate.py --config configs/image.toml --split eval
uv run scripts/evaluate.py --config configs/fusion.toml --split eval
uv run scripts/audio_gmm_pipeline.py --mode map_free
uv run scripts/audio_gmm_hybrid.py
uv run scripts/fix_vad_empty.py
uv run scripts/fusion_hybrid_image.py
uv run scripts/fusion_3way.py
```

All randomness is seeded. Reproduction is deterministic from a fixed
seed.
