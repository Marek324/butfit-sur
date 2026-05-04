---
title: "SUR 2025/2026 Project"
author:
 - "Marek Hric (xhricma00)"
 - "Jakub Komanický (xkomanj00)"
date: "2026-05-04"
geometry: margin=2cm
fontsize: 10pt
---

# 1. Data and Augmentation

The supplied splits are session-disjoint for the target (train holds
*m431* sessions 01–02, dev holds session 03) and speaker-disjoint for
the impostors (13 speakers in train, 6 distinct speakers in dev),
which pre-stresses generalisation to unseen identities and an unseen
recording session of the target. The remainder of this report traces
each per-modality system from the syllabus baseline to its production
form by changing one design choice per step, so individual metric
deltas are attributable to specific decisions.

Augmentation is performed once, offline (`scripts/augment_dataset.py`),
producing `dataset/*_aug` directories that the trainer enumerates as
ordinary samples. Audio augmentation applies time stretch (0.9–1.1×),
gain (±6 dB), polarity flip ($p=0.5$), peaking equalisation (±3 dB at
200–4000 Hz) and additive white Gaussian noise (SNR 15–30 dB). Image
augmentation applies horizontal flip ($p=0.5$), rotation (±10°),
translation (±5 %), brightness and contrast scaling (0.85–1.15×) and
per-pixel WGN ($\sigma\in[0,0.04]$). Vertical flip and hue rotation are
excluded because faces are not vertically symmetric and skin tone is
identity-relevant. The augmentation count is 15× per target sample and
5× per non-target, an explicit asymmetry, since the target class has
only 30 base utterances total. Class imbalance is handled at the loss
level via `class_weight='balanced'`, so the
trained classifier intercept satisfies $P(\text{target}\mid x)=0.5$ at
logit zero regardless of the empirical training ratio.

# 2. Evaluation Metrics

Four quantities are reported throughout. **AUC** is the primary
rank statistic, threshold-free and robust to class imbalance; most
variant-selection decisions (audio feature subsets, image
regularisation, GMM-UBM mixture counts) were made on AUC.
**Equal Error Rate (EER) and the EER threshold** summarise the
symmetric-cost operating point under prior 0.5; drift of the EER
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

Image features are 9-bin Histograms of Oriented Gradients
[1] over 8×8-pixel cells with 2×2-cell L2-Hys normalised
blocks on 80×80 grayscale crops, yielding 2916-d vectors. A balanced
logistic regression with L2 regularisation ($C=0.1$) classifies them.
Hard decisions threshold $\mathrm{decision\_function}$ at zero. The
progression from the syllabus baseline isolates three design changes:

| Step | Description | AUC | EER |
|------|-------------|----:|----:|
| 1 | Baseline: pixels → PCA(D=21) → Gaussian LDA + 2 IRLS | 0.943 | 20.00 % |
| 2 | HOG → PCA(D=21) → Gaussian LDA + 2 IRLS | 0.977 | 9.17 % |
| 3 | HOG (full 2916-d) → balanced LogReg, $C=0.1$, no augmentation | 0.903 | 21.67 % |
| 4 | + offline augmentation (production) | 1.000 | 0.00 % |

Step 1→2 isolates the HOG vs. raw-pixel feature gain at fixed
dimensionality. Step 2→3 is a *regression*: dropping PCA in favour of
full-dimensional HOG with L2 leaves the threshold-zero hard rule
mis-calibrated at only 10 target training images. Step 3→4 shows that
augmentation is what makes the high-dimensional architecture viable;
full-dimensional + L2 in isolation is worse than the PCA-bridged
baseline classifier of step 2.

# 4. Audio System

Both audio systems share the same frontend: an energy + zero-crossing-rate
voice activity detector (VAD), 13 mel-frequency cepstral coefficients [2]
at 25 ms / 10 ms framing with per-utterance cepstral mean–variance
normalisation, and probabilistic-YIN F$_0$ estimation [3] over voiced
frames in 60–400 Hz. From the F$_0$ track and frame energies
we compute five prosodic statistics ($\log F_0$ mean and standard
deviation, jitter, shimmer, voicing ratio).

**Path A, aggregated features and logistic regression.** The 5
prosodic statistics and the 13 MFCC means form an 18-d utterance-level
vector classified by balanced LogReg ($C=1.0$). This is shipped as
`audio_logreg.txt`.

**Path B, Gaussian Mixture Model with universal background, MAP-free.**
Following [4] we fit a 16-mixture diagonal-covariance
universal background model (UBM) on non-target frames and a 4-mixture
diagonal-covariance target GMM on *m431* frames, *independently* by EM.
We chose MAP-free over MAP adaptation because ~32 k augmented target
frames are plentiful at the per-component level ($\alpha_k\to 1$
regardless of relevance factor), and a relevance-factor sweep over
$\tau\in\{5, 10, 20, 100, 500, 2000, 10000\}$ produced an AUC plateau
of 0.98 with the threshold-zero hard rule miscalibrated (EER threshold
drifted to $+0.25$). The per-utterance score is the frame-averaged
log-likelihood ratio. We did not attempt i-vector front-end factor
analysis [5], the standard upgrade over GMM-UBM, because it requires
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
real speech, not white noise (median voiced fraction 0.60); the
sentinel default was discarding genuine information. The same
pathology contaminates training: 121 of 1112 augmented training
utterances also extract to zero vectors, with the target class
disproportionately affected (81 zero-frame target augmentations vs. 40
non-target), so the model also *learned* a small "zero $\Rightarrow$
target" correlation as a class-asymmetric augmentation artefact.

**Hybrid VAD-on / VAD-off fallback.** The fix is a routing scheme
(`scripts/audio_gmm_hybrid.py`): an auxiliary 16+4 GMM-UBM is trained
on the same MFCC frames *without* applying VAD; at scoring time the
production VAD-on model is consulted first, and the no-VAD model takes
over only when VAD has trimmed every frame. The hybrid is shipped as
`audio_gmm_hybrid.txt` and short-circuits both the eval-time sentinel
defaults and the train-time spurious correlation at scoring.

The audio progression covers both clean and stress development splits:

| Step | Description | Clean AUC | Stress AUC |
|------|-------------|----------:|-----------:|
| 1 | Baseline: course MFCC + 2 GMMs M=30 (no VAD, no CMVN) | 0.978 | 0.799 |
| 2 | + frame-energy VAD + per-utt CMVN, same 2 GMMs M=30 | 1.000 | 0.730 |
| 3 | Production GMM-UBM (MAP-free, sentinel for VAD-empty) | 0.997 | 0.463 |
| 4 | Hybrid VAD-on/off (production) | 0.997 | **0.870** |

VAD and CMVN saturate clean dev (steps 1→2) but degrade the stress
split because they introduce the failure mode that step 4 must repair.
The hybrid recovers the baseline's stress robustness without
sacrificing clean-dev performance.

# 5. Multimodal Fusion

Fusion is performed at the score level. For each per-modality system,
dev-set scores are z-normalised; a balanced logistic regression is
fitted on the concatenated z-scores and the ground-truth labels; the
classifier is then applied to z-normalised eval scores. Score-level was
preferred to feature-level because the audio (18-d) and image (2916-d)
spaces are extremely imbalanced: stacking them with ~1100 training
samples invites the curse of dimensionality, whereas a 2- or 3-feature
fusion logreg has too few parameters to overfit the 70-sample dev split.
We submit three variants: `fusion_logreg.txt` over $(z(\text{audio\_logreg}),
z(\text{image}))$; `fusion_hybrid_image.txt` over
$(z(\text{audio\_gmm\_hybrid}), z(\text{image}))$; and `fusion_3way.txt`
over $(z(\text{audio\_logreg}), z(\text{audio\_gmm\_hybrid}),
z(\text{image}))$. The three-way variant tests whether the two audio
paths capture complementary signal, since aggregated 18-d features and
frame-level GMM likelihoods are structurally different objects.

# 6. Validation Strategy

The clean `*_dev` split (n=70) is the primary diagnostic. To probe
generalisation under input degradation we built a *stress* dev: the
same dev samples re-augmented with a *disjoint* envelope (gain $-15$
to $-8$ dB, SNR 3–13 dB, peaking EQ $-7$ to $-4$ dB at 4500–7500 Hz,
time-stretch 0.78–0.88, image brightness 0.45–0.70, per-pixel noise
$\sigma\in[0,0.10]$), yielding 50 target / 120 non-target stress samples.
Disjointness converts each stress sample from interpolation to
extrapolation under the trained model.

We also ran a 2-fold leave-one-target-session-out session-grouped
cross-validation (`scripts/cv_session_check.py`) on the training pool.
Image AUC stays at 0.982 across folds; audio AUC drops to 0.789,
confirming image features are session-invariant and audio is
session-sensitive, which motivates using *both* available target
sessions (01 and 02) for production training.

The 50/50-prior decision rule is enforced in two places: every
classifier sets `class_weight='balanced'` (equalising effective
per-class loss mass during training, so the trained intercept
satisfies $P(\text{target}\mid x)=0.5$ at logit zero); and every config
sets `output.hard_threshold_logit = 0.0`, so the hard decision is
$\mathrm{score}\geq 0$. This implements the symmetric-cost
detection-cost operating point [6].

# 7. Results

The submission ships one image system, two audio systems and three
fusion variants, 6 result files in total.

**Clean dev** (n = 70: 10 target / 60 non-target):

| System | AUC | EER | TP | FP | FN | TN |
|--------|----:|----:|---:|---:|---:|---:|
| image_logreg | 1.000 | 0.00 % | 10 | 1 | 0 | 59 |
| audio_logreg | 0.983 | 3.33 % | 7 | 0 | 3 | 60 |
| audio_gmm_hybrid | 0.997 | 1.67 % | 9 | 0 | 1 | 60 |
| fusion_logreg | 1.000 | 0.00 % | 10 | 0 | 0 | 60 |
| fusion_hybrid_image | 1.000 | 0.00 % | 10 | 0 | 0 | 60 |
| fusion_3way | 1.000 | 0.00 % | 10 | 0 | 0 | 60 |

**Stress dev** (n = 170: 50 target / 120 non-target):

| System | AUC | EER | TP | FP | FN | TN |
|--------|----:|----:|---:|---:|---:|---:|
| image_logreg | 0.607 | 40.00 % | 35 | 60 | 15 | 60 |
| audio_logreg | 0.521 | 47.08 % | 50 | 113 | 0 | 7 |
| audio_gmm_hybrid | 0.870 | 24.08 % | 47 | 69 | 3 | 51 |
| fusion_logreg | 0.625 | 39.17 % | 50 | 111 | 0 | 9 |
| fusion_hybrid_image | 0.875 | 20.42 % | 50 | 107 | 0 | 13 |
| fusion_3way | 0.860 | 21.83 % | 35 | 15 | 15 | 105 |

On the unlabelled eval set (n = 1346), predicted-target counts under
the threshold-zero rule are 127 (image_logreg), 153 (audio_logreg),
139 (audio_gmm_hybrid), 142 (fusion_logreg), 178 (fusion_hybrid_image)
and 139 (fusion_3way). DET curves [6] are saved at
`artifacts/curves*.png`; we prefer DET over ROC because its
log-deviate axes spread the high-performance region that AUC
saturation collapses into the corner of the ROC.

All three fusion variants saturate clean dev. Stress dev separates
them clearly: `fusion_hybrid_image` maximises AUC (0.875);
`fusion_3way` is the only system whose stress confusion matrix is
balanced (precision 0.70 versus 0.31 for the alternatives), trading a
small AUC decrement for substantial precision gain. Under an operating
point that penalises false positives more strongly than the
assignment's symmetric costs, `fusion_3way` would be the strongest
single submission.

# 8. Discussion

The most surprising finding is the *clean–stress inversion* in the
audio progression: the simple syllabus baseline (no VAD, no CMVN)
handles the stress split substantially better than the production
preprocessed pipeline does. The mechanism is that the production VAD
trims every frame on heavily attenuated audio (161 of 170 stress
samples and 79 of 1346 eval samples produce all-zero feature vectors);
without VAD, the attenuated signal is still scored on its remaining
content. The hybrid fallback recovers the baseline's stress robustness
on the GMM-UBM path while preserving clean-dev saturation. We did not
attempt to remove or retune the VAD on the LogReg path because
disabling it costs ~35 AUC points on clean dev, and a routing fix
restricted to the small known-degenerate slice is a smaller
intervention than a full pipeline retune. A from-scratch convolutional
network was ruled out for reasons analogous to i-vectors: a small CNN
(~50 k parameters) needs an order of magnitude more target examples
than the 320 augmented samples we have.

# 9. Reproduction

```bash
uv run scripts/augment_dataset.py --config configs/augment_audio.toml
uv run scripts/augment_dataset.py --config configs/augment_image.toml
uv run scripts/train.py --config configs/audio_baseline.toml
uv run scripts/train.py --config configs/image_baseline.toml
uv run scripts/train.py --config configs/fusion_baseline.toml
uv run scripts/evaluate.py --config configs/audio_baseline.toml --split eval
uv run scripts/evaluate.py --config configs/image_baseline.toml --split eval
uv run scripts/evaluate.py --config configs/fusion_baseline.toml --split eval
uv run scripts/audio_gmm_pipeline.py --mode map_free
uv run scripts/audio_gmm_hybrid.py
uv run scripts/fix_vad_empty.py
uv run scripts/fusion_hybrid_image.py
uv run scripts/fusion_3way.py
```

All randomness is seeded; reproduction is deterministic from a fixed
seed.

# References

1. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, San Diego, USA, 2005, pp. 886–893. doi:10.1109/CVPR.2005.177.

2. S. Davis and P. Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences," *IEEE Transactions on Acoustics, Speech and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980. doi:10.1109/TASSP.1980.1163420.

3. M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, Florence, Italy, 2014, pp. 659–663. doi:10.1109/ICASSP.2014.6853678.

4. D. A. Reynolds, T. F. Quatieri, and R. B. Dunn, "Speaker Verification Using Adapted Gaussian Mixture Models," *Digital Signal Processing*, vol. 10, no. 1–3, pp. 19–41, 2000. doi:10.1006/dspr.1999.0361.

5. N. Dehak, P. J. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-End Factor Analysis for Speaker Verification," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 19, no. 4, pp. 788–798, 2011. doi:10.1109/TASL.2010.2064307.

6. A. Martin, G. Doddington, T. Kamm, M. Ordowski, and M. Przybocki, "The DET Curve in Assessment of Detection Task Performance," in *Proc. 5th European Conference on Speech Communication and Technology (Eurospeech)*, Rhodes, Greece, 1997, pp. 1895–1898.
