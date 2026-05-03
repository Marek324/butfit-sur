# Proposed experiments — 2026-05-03

What we've actually run lives in `configs/experiments/` + the `run_experiments.py` sweep. This entry catalogues additional ideas, prioritized for the time budget remaining before submission. All comparisons should be on **stress dev**, since clean dev saturates and isn't diagnostic.

## Tier 1 — quick wins

These are config-only changes the existing scaffold supports today:

1. **Image PCA reduction.** `image.features.pca_components > 0` is a config field but no implementation. Wire it up: fit PCA on training features, transform train+dev+eval. Targets the same "low-capacity → better stress" axis as `hog_coarser`, but keeps the rich HOG signal and just reduces the input dim presented to the classifier. Implementing requires fitting + persisting the PCA basis with the model.

2. **L1-regularized logreg.** `penalty = "l1"` (already supported in config) — feature selection via sparsity. Identifies which HOG cells / MFCC means are actually doing work.

3. **Class-balanced augmentation match.** Currently `n_per_target = 15`, `n_per_nontarget = 5` — non-targets are ~83% noisy, targets ~94%. Set `n_per_nontarget = 15` for true symmetry. Cost: ~30 min audio extraction once. May help stress-dev image too.

4. **Fusion `score_normalize = "none"` ablation.** Current default is `"zscore"`. Test whether normalization is actually helping or just absorbing slack.

## Tier 2 — small refactors required

5. **Δ-MFCC / ΔΔ-MFCC means.** Stack first/second derivatives of MFCC means in `audio_features.py` — captures temporal dynamics of timbre. Could help under noise (deltas are mean-removed, so cancel some additive bias). Maybe 5 base + 5 Δ + 5 ΔΔ MFCC means → 15 spectral features, 20-d total.

6. **MFCC standard deviations alongside means.** Currently we take per-utterance MFCC means; the std (variability) of MFCC over voiced frames is a separate descriptor. Adds ~13 features.

7. **Spectral entropy / centroid means.** Two more cheap utterance-level descriptors. Both robust to additive noise (entropy normalizes by total power).

8. **CV-OOF fusion training.** `fusion.train_on = "cv_oof"` raises NotImplementedError. Wire it up: K-fold per-modality models, OOF scores from each fold form the fusion training set. Removes the dev-leak bias that makes fusion's reported metrics optimistic.

## Tier 3 — bigger swings

9. **GMM-UBM audio.** Frame-level MFCC + ΔMFCC, fit a small UBM on non-target train, MAP-adapt a target model from it; score = log-likelihood ratio. The textbook approach, more expressive than utterance-level descriptors. Bigger refactor — frame-level features and a new classifier path.

10. **Small CNN on image patches.** Train from scratch on 80×80 with heavy augmentation. Forbidden in spirit ("no pretrained models") but a from-scratch CNN is allowed. Risk: 1112 samples may not be enough; benefit: the eval set is "noisy/different conditions" and a CNN with strong augmentation might generalize where HOG won't.

11. **Speaker-specific modeling.** Instead of one binary classifier, model the m431 acoustic distribution explicitly (e.g., diagonal-covariance Gaussian over the 10-d feature) and score by likelihood-ratio against the empirical non-target distribution. Closer to the GMM-UBM idea but cheap.

## What I'd prioritize given time pressure

If there's an evening of compute left:
- (3) Class-balanced augmentation match — cheap, addresses a real asymmetry already identified
- (5) Δ-MFCC means — meaningful new signal at low cost
- (1) Image PCA — addresses generalization-vs-saturation tradeoff cleanly

If the audio system needs rescuing (stress AUC is 0.53, near chance):
- (9) GMM-UBM as a fallback to the logreg
- (5)+(6)+(7) feature expansion

If documentation time is the main scarcity, skip all of these and write the doc — the existing baselines work, and they're defensible.

## What I'd NOT prioritize

- Hyperparameter grid search beyond a few discrete C values. Limited data, large variance.
- Deep ensembles / stacking. Overkill at this scale.
- Transfer learning from any pretrained image feature extractor. **Forbidden by the assignment.**
