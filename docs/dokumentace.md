---
title: "Person Verification for Subject m431 from Face and Voice"
subtitle: "SUR 2025/2026 Project Documentation"
author: "Marek (login: hrmajo1)"
date: "2026-05-04"
geometry: margin=2cm
fontsize: 10pt
---

# Abstract

We build a person-verification detector for subject *m431* from face crops and
voice recordings under the assignment's no-external-data, no-pretrained-models
constraint. The pipeline combines classical feature extractors — Histogram of
Oriented Gradients [@dalal2005hog] for images, mel-frequency cepstral
coefficients [@davis1980mfcc] with prosodic statistics for audio — with
balanced linear classifiers calibrated to the prior-0.5 decision rule. From
the syllabus baselines (PCA + Gaussian LDA for images; two-GMM scoring for
audio) we trace each system to its production form by changing one design
choice per step. A disjoint-augmentation *stress* development split exposed a
voice activity detection failure on attenuated audio; a hybrid VAD-on / VAD-off
GMM-UBM [@reynolds2000gmmubm] system patches the failure and recovers stress
robustness without regressing the saturated clean development split. The
submission ships six result files: one image, two audio, and three fusion
variants over their score-level combinations.

# 1. Data and Augmentation

The corpus is 80×80 RGB face crops (PNG) and 16-kHz monophonic recordings
(WAV). Target train holds 20 paired stems (sessions 01–02 of *m431*); target
dev holds 10 paired stems (session 03). Non-target train has 132 stems from
13 speakers; non-target dev has 60 stems from a *disjoint* set of 6 speakers.
The split is therefore session-disjoint for *m431* and speaker-disjoint for
the impostors, pre-stressing generalisation to unseen identities.

Augmentation is offline (`scripts/augment_dataset.py`), producing
`dataset/*_aug` directories enumerated as ordinary samples at training. Audio
augmentation applies time stretch (0.9–1.1×), gain (±6 dB), polarity flip
($p=0.5$), peaking EQ (±3 dB at 200–4000 Hz) and additive WGN
(SNR 15–30 dB). Image augmentation applies horizontal flip ($p=0.5$),
rotation (±10°), translation (±5 %), brightness and contrast scaling
(0.85–1.15×) and per-pixel WGN ($\sigma\in[0,0.04]$). Vertical flip and hue
rotation are excluded because faces are not vertically symmetric and skin
tone is identity-relevant. The augmentation count is 15× per target sample
and 5× per non-target — an explicit asymmetry, since the target class has
only 30 base utterances total. Class imbalance is handled at the loss level
via `class_weight='balanced'` [@pedregosa2011scikit], so the trained
classifier intercept satisfies $P(\text{target}\mid x)=0.5$ at logit zero
regardless of the empirical training ratio.

# 2. Methods

## 2.1 Image: HOG + regularised logistic regression

Image features are 9-bin HOG [@dalal2005hog] over 8×8-pixel cells with
2×2-cell L2-Hys normalised blocks on 80×80 grayscale crops, yielding 2916-d
vectors. A balanced LogReg ($C=0.1$, L2) classifies them. Hard decisions
threshold $\mathrm{decision\_function}$ at zero. The progression from the
syllabus baseline isolates three design changes:

| Step | Description | AUC | EER |
|------|-------------|----:|----:|
| 1 | Baseline: pixels → PCA(D=21) → Gaussian LDA + 2 IRLS                   | 0.943 | 20.00 % |
| 2 | HOG → PCA(D=21) → Gaussian LDA + 2 IRLS                                | 0.977 |  9.17 % |
| 3 | HOG (full 2916-d) → balanced LogReg, $C=0.1$, no augmentation         | 0.903 | 21.67 % |
| 4 | + offline augmentation (production)                                    | 1.000 |  0.00 % |

Step 1→2 isolates the HOG vs. raw-pixel feature gain at fixed dimensionality.
Step 2→3 is a *regression*: dropping PCA in favour of full-dimensional HOG +
L2 leaves the threshold-zero hard rule mis-calibrated with only 10 target
training images. Step 3→4 shows that augmentation is what makes the
high-dimensional architecture viable; full-dimensional + L2 in isolation is
worse than the PCA-bridged baseline classifier.

## 2.2 Audio: two complementary paths

Both audio systems share the same frontend: an energy + zero-crossing-rate
voice activity detector (VAD), 13 MFCCs at 25 ms / 10 ms framing via
`librosa` [@mcfee2015librosa] with per-utterance cepstral
mean–variance normalisation, and probabilistic-YIN F$_0$ estimation
[@mauch2014pyin] over voiced frames in 60–400 Hz. From the F$_0$ track
we compute five prosodic statistics ($\log F_0$ mean and standard
deviation, jitter, shimmer, voicing ratio).

**Path A — aggregated features + LogReg.** The 5 prosodic statistics and
the 13 MFCC means form an 18-d utterance-level vector classified by
balanced LogReg ($C=1.0$). Submitted as `audio_logreg.txt`.

**Path B — GMM-UBM, MAP-free.** Following [@reynolds2000gmmubm], we fit
a 16-mixture diagonal-cov universal background model on non-target
frames and a 4-mixture diagonal-cov target GMM on *m431* frames,
*independently* by EM. We chose MAP-free over MAP adaptation because
~32 k augmented target frames are plentiful at the per-component level
($\alpha_k\to 1$ regardless of relevance factor), and a $\tau$-sweep over
$\{5, 10, 20, 100, 500, 2000, 10000\}$ produced an AUC plateau of 0.98
with the threshold-zero hard rule miscalibrated (EER threshold drifted
to $+0.25$). Per-utterance score is the frame-averaged log-likelihood
ratio.

**The VAD-empty pathology.** Auditing eval revealed 79 of 1346 samples
(5.9 %) producing identical audio score $\approx +1.535$. These extract
to all-zero feature vectors: under heavy gain attenuation or short
voiced segments, the energy-relative VAD trims every frame, leaving
prosodic and CMVN-MFCC statistics at their zero-fallback. The trained
classifier maps this zero input to a bias-driven default that happened
to be positive, mislabelling all 79 as target. Spectral-flatness and
voiced-fraction analysis confirmed all 79 are real speech, not white
noise (median voiced fraction 0.60); the sentinel default was discarding
genuine information.

**Hybrid VAD-on / VAD-off fallback.** The fix is a routing scheme
(`scripts/audio_gmm_hybrid.py`): an auxiliary 16+4 GMM-UBM is trained on
the same MFCC frames *without* applying VAD; at scoring time the
production VAD-on model is consulted first, and the no-VAD model takes
over only when VAD has trimmed every frame. Submitted as
`audio_gmm_hybrid.txt`.

| Step | Description | Clean AUC | Stress AUC |
|------|-------------|----------:|-----------:|
| 1 | Baseline: course MFCC + 2 GMMs M=30 (no VAD, no CMVN)              | 0.978 | 0.799 |
| 2 | + frame-energy VAD + per-utt CMVN, same 2 GMMs M=30                | 1.000 | 0.730 |
| 3 | Production GMM-UBM (MAP-free, sentinel for VAD-empty)              | 0.997 | 0.463 |
| 4 | Hybrid VAD-on/off (production)                                     | 0.997 | **0.870** |

The progression makes a non-obvious finding visible: VAD and CMVN
saturate clean dev (1→2) but degrade stress dev because they introduce
the failure mode that step 4 must repair. The hybrid recovers the
baseline's stress robustness without sacrificing clean-dev performance.

## 2.3 Multimodal fusion

Fusion is at the score level. Per-modality dev scores are z-normalised; a
balanced LogReg is fitted on the concatenated z-scores and dev labels, and
applied to z-normalised eval scores at test time. Score-level was preferred
to feature-level because the audio (18-d) and image (2916-d) feature spaces
are extremely imbalanced: stacking them with ~1100 training samples invites
the curse of dimensionality, whereas a 2- or 3-feature fusion logreg has
too few parameters to overfit the 70-sample dev split. We submit:

- `fusion_logreg.txt`: $(z(\text{audio\_logreg}), z(\text{image}))$.
- `fusion_hybrid_image.txt`: $(z(\text{audio\_gmm\_hybrid}), z(\text{image}))$.
- `fusion_3way.txt`: $(z(\text{audio\_logreg}), z(\text{audio\_gmm\_hybrid}), z(\text{image}))$.

# 3. Validation Strategy

The clean `*_dev` split (n=70) is the primary diagnostic. To probe
generalisation under input degradation we built a *stress* dev: same dev
samples re-augmented with a *disjoint* envelope (gain $-15$ to $-8$ dB,
SNR 3–13 dB, peaking EQ $-7$ to $-4$ dB at 4500–7500 Hz, time-stretch
0.78–0.88, image brightness 0.45–0.70, per-pixel noise $\sigma\in[0,0.10]$),
yielding 50 target / 120 non-target stress samples. Disjointness converts
each stress sample from interpolation to extrapolation under the trained
model.

We also ran a 2-fold leave-one-target-session-out session-grouped CV
diagnostic (`scripts/cv_session_check.py`) on the training pool. Image
holds AUC = 0.982 across folds; audio drops to 0.789 — confirming image
features are session-invariant and audio is session-sensitive, motivating
the use of *both* available target sessions for production training.

The 50/50-prior decision rule is enforced in two places: (i)
`class_weight='balanced'` in every classifier, equalising effective
per-class loss mass during training; (ii) `output.hard_threshold_logit
= 0.0` in every config, so the hard decision is `score >= 0`. This
implements the symmetric-cost detection-cost operating point
[@martin1997det].

# 4. Results

**Clean dev** (n = 70):

| System | AUC | EER | TP | FP | FN | TN |
|--------|----:|----:|---:|---:|---:|---:|
| image_logreg               | 1.000 |  0.00 % | 10 | 1 | 0 | 59 |
| audio_logreg               | 0.983 |  3.33 % |  7 | 0 | 3 | 60 |
| audio_gmm_hybrid           | 0.997 |  1.67 % |  9 | 0 | 1 | 60 |
| fusion_logreg              | 1.000 |  0.00 % | 10 | 0 | 0 | 60 |
| fusion_hybrid_image        | 1.000 |  0.00 % | 10 | 0 | 0 | 60 |
| fusion_3way                | 1.000 |  0.00 % | 10 | 0 | 0 | 60 |

**Stress dev** (n = 170):

| System | AUC | EER | TP | FP | FN | TN |
|--------|----:|----:|---:|---:|---:|---:|
| image_logreg               | 0.607 | 40.00 % | 35 |  60 | 15 |  60 |
| audio_logreg               | 0.521 | 47.08 % | 50 | 113 |  0 |   7 |
| audio_gmm_hybrid           | 0.870 | 24.08 % | 47 |  69 |  3 |  51 |
| fusion_logreg              | 0.625 | 39.17 % | 50 | 111 |  0 |   9 |
| fusion_hybrid_image        | 0.875 | 20.42 % | 50 | 107 |  0 |  13 |
| fusion_3way                | 0.860 | 21.83 % | 35 |  15 | 15 | 105 |

**Eval set** (n = 1346, no labels): predicted-target counts are 127
(image_logreg), 153 (audio_logreg), 139 (audio_gmm_hybrid), 142
(fusion_logreg), 178 (fusion_hybrid_image), 139 (fusion_3way). DET curves
[@martin1997det] at `artifacts/curves*.png` are preferred to ROC because
their log-deviate axes spread the high-performance region that AUC
saturation collapses into a corner.

All three fusion variants saturate clean dev. Stress dev separates them
clearly: `fusion_hybrid_image` maximises AUC (0.875); `fusion_3way` is
the only system whose stress confusion matrix is balanced (precision 0.70
versus 0.31 for the alternatives), trading a small AUC decrement for
substantial precision gain. Under an operating point that penalises false
positives more than the assignment's symmetric costs, `fusion_3way` is the
strongest single submission.

# 5. Discussion

The most surprising finding is the *clean–stress inversion* in the audio
progression: the simple syllabus baseline (no VAD, no CMVN) handles stress
substantially better than the production preprocessed pipeline. The
mechanism is that the production VAD trims every frame on heavily
attenuated audio (161 of 170 stress samples and 79 of 1346 eval samples
produce all-zero feature vectors); without VAD, the attenuated signal is
still scored on its remaining content. The hybrid fallback recovers the
baseline's stress robustness and additionally fixes a subtle
training-data contamination — 121 of 1112 augmented training utterances
also extract to zero vectors, with the target class disproportionately
affected (81 zero-frame target augmentations vs. 40 non-target), causing
the model to learn a small "zero $\Rightarrow$ target" correlation as a
class-asymmetric augmentation artefact. The hybrid short-circuits this
spurious association at scoring time.

Several simplifications were forced by the data scale. We did not attempt
i-vector front-end factor analysis [@dehak2011ivector] — the standard
upgrade over GMM-UBM — because it requires hundreds of background speakers
for the total-variability matrix to be estimable, whereas our pool has 19
distinct non-target speakers. A from-scratch convolutional network was
ruled out for analogous reasons: a small CNN (~50 k parameters) needs
order-of-magnitude more target examples than our 320 augmented samples.

# 6. Reproduction

```bash
uv run scripts/augment_dataset.py --config configs/augment_audio.toml
uv run scripts/augment_dataset.py --config configs/augment_image.toml
uv run scripts/train.py    --config configs/audio_baseline.toml
uv run scripts/train.py    --config configs/image_baseline.toml
uv run scripts/train.py    --config configs/fusion_baseline.toml
uv run scripts/evaluate.py --config configs/audio_baseline.toml  --split eval
uv run scripts/evaluate.py --config configs/image_baseline.toml  --split eval
uv run scripts/evaluate.py --config configs/fusion_baseline.toml --split eval
uv run scripts/audio_gmm_pipeline.py --mode map_free
uv run scripts/audio_gmm_hybrid.py
uv run scripts/fix_vad_empty.py
uv run scripts/fusion_hybrid_image.py
uv run scripts/fusion_3way.py
```

All randomness is seeded; reproduction is deterministic from a fixed seed.

# References

```bibtex
@article{davis1980mfcc,
  author={Davis, S. and Mermelstein, P.},
  title={Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences},
  journal={IEEE Transactions on Acoustics, Speech and Signal Processing},
  volume={28}, number={4}, pages={357--366}, year={1980},
  doi={10.1109/TASSP.1980.1163420}
}
@inproceedings{martin1997det,
  author={Martin, A. and Doddington, G. and Kamm, T. and Ordowski, M. and Przybocki, M.},
  title={The {DET} Curve in Assessment of Detection Task Performance},
  booktitle={Proc. 5th European Conference on Speech Communication and Technology},
  pages={1895--1898}, year={1997},
  doi={10.21437/Eurospeech.1997-504}
}
@article{reynolds2000gmmubm,
  author={Reynolds, D. A. and Quatieri, T. F. and Dunn, R. B.},
  title={Speaker Verification Using Adapted {G}aussian Mixture Models},
  journal={Digital Signal Processing},
  volume={10}, number={1--3}, pages={19--41}, year={2000},
  doi={10.1006/dspr.1999.0361}
}
@inproceedings{dalal2005hog,
  author={Dalal, N. and Triggs, B.},
  title={Histograms of Oriented Gradients for Human Detection},
  booktitle={Proc. IEEE Conf.\ Computer Vision and Pattern Recognition (CVPR)},
  pages={886--893}, year={2005},
  doi={10.1109/CVPR.2005.177}
}
@article{pedregosa2011scikit,
  author={Pedregosa, F. and others},
  title={Scikit-learn: Machine Learning in {P}ython},
  journal={Journal of Machine Learning Research},
  volume={12}, pages={2825--2830}, year={2011}
}
@article{dehak2011ivector,
  author={Dehak, N. and Kenny, P. J. and Dehak, R. and Dumouchel, P. and Ouellet, P.},
  title={Front-End Factor Analysis for Speaker Verification},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  volume={19}, number={4}, pages={788--798}, year={2011},
  doi={10.1109/TASL.2010.2064307}
}
@inproceedings{mauch2014pyin,
  author={Mauch, M. and Dixon, S.},
  title={{pYIN}: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions},
  booktitle={Proc. IEEE Int.\ Conf.\ Acoustics, Speech and Signal Processing (ICASSP)},
  pages={659--663}, year={2014},
  doi={10.1109/ICASSP.2014.6853678}
}
@inproceedings{mcfee2015librosa,
  author={McFee, B. and Raffel, C. and Liang, D. and Ellis, D. P. W. and McVicar, M. and Battenberg, E. and Nieto, O.},
  title={{librosa}: Audio and Music Signal Analysis in {Python}},
  booktitle={Proc. 14th Python in Science Conference (SciPy)},
  pages={18--24}, year={2015},
  doi={10.25080/Majora-7b98e3ed-003}
}
```
