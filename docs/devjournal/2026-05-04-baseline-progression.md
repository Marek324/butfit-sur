# Baseline → production progression - 2026-05-04

Reproduced the teacher's IKR-syllabus baselines verbatim and ran a
minimal sequence of intermediate experiments toward the production
systems. The point: each delta should be attributable to one design
choice. Results in `artifacts/baselines_metrics.json` and
`artifacts/baseline_{audio,image}_step{2,3}.json`. Code lives in
`scripts/baselines/`.

## Image

| Step | Features | Classifier | Reg / aug | AUC | EER |
|------|----------|------------|-----------|----:|----:|
| 1 | pixels → PCA(D=21) | Gaussian LDA + 2 IRLS | none | 0.943 | 20.00 % |
| 2 | HOG (2916-d)       | Gaussian LDA + 2 IRLS | none | *singular* | — |
| 3 | HOG (2916-d)       | balanced LogReg       | L2 C=0.1, **no aug** | 0.903 | 21.67 % |
| 4 | HOG (2916-d)       | balanced LogReg       | L2 C=0.1 + augmentation | **1.000** | **0.00 %** |

Three lessons in three deltas:

- **Step 1 → 2**: Replacing pixels with HOG without changing the classifier breaks the pipeline outright. Pooled covariance over a 2916-d feature space with 152 training samples is rank-deficient, IRLS Hessian goes singular, generative LDA's `inv()` returns garbage and AUC drops to 0.668. The information is in HOG, but the unregularised classifier can't get at it. *You cannot independently change features and classifier when the dimensionality changes by 30×.*
- **Step 2 → 3**: L2 regularisation rescues the inverse problem. Ranking quality returns (AUC 0.903), but with only 10 target training images the threshold-0 hard rule lands far off the score-distribution centre — TP = 2/10. The boundary is roughly placed; its offset is wrong because there's no calibration data on the target side.
- **Step 3 → 4**: Offline augmentation (rotation, brightness/contrast, per-pixel noise, horizontal flip) supplies target variation. The dev AUC saturates at 1.000 and TP becomes 10/10 with one FP. Augmentation here is doing two jobs at once: more target samples for the boundary, and richer in-class variation that pulls the decision threshold to the right place.

The 2 → 3 → 4 sequence is the main story for image: HOG features need *both* explicit regularisation *and* enough target data (via augmentation) to be useful at this dimensionality. Either one alone is insufficient.

## Audio

| Step | Features | Classifier | AUC | EER |
|------|----------|------------|----:|----:|
| 1 | course MFCC (no VAD, no CMVN), 13-d frames | 2 GMMs M = 30, 40 EM iters | 0.978 | 8.33 % |
| 2 | course MFCC + frame-energy VAD + per-utt CMVN | same 2 GMMs M = 30, 40 EM iters | **0.998** | **0.83 %** |
| 3 | librosa CMVN-MFCC + production VAD + augmentation | 16-mix UBM + 4-mix target, MAP-free, frame-mean LLR | 0.997 | 1.67 % |

Two lessons:

- **Step 1 → 2**: Adding VAD and CMVN inside the same 2-GMM-per-class architecture takes the audio baseline from AUC 0.978 to 0.998. **Preprocessing carries almost the entire delta.** MFCC means without VAD are dragged toward silence and boundary noise; without CMVN, level and channel differences between sessions look discriminative when they shouldn't. The model didn't need to change.
- **Step 2 → 3**: Switching to a 16 + 4 GMM-UBM with frame-mean LLR (production) does not improve clean dev — it slightly regresses EER (0.83 → 1.67 %). The architecture change isn't justified by clean dev metrics; it's justified by (a) compute economy during the variant sweeps (fewer mixtures, sklearn EM is faster than the course's pure-Python EM), and (b) honest geometry of a 1-vs-many verification task: one target speaker inside a wider impostor distribution argues for an asymmetric model.

The progression makes a non-obvious claim: for our data, **course MFCC + simple energy VAD + CMVN + 2 GMMs M = 30** is competitive with or even slightly better than our production GMM-UBM on clean dev. The honest defence is that the differences are within the noise of a 70-sample dev split, and the production system was chosen for properties that don't show up on clean dev — frame-budget stability, eval-time speed, and the assumption that eval would contain noise the stress dev was built to probe (which itself revealed the VAD-empty pathology, see §7.3).

## What didn't make it into the table

Decisions whose delta on clean dev is zero or negative but justified elsewhere:

- *Class-symmetric augmentation* (`n_per_nontarget = 5`) — prevents the "noisy ⇒ target" shortcut. Only matters when target-only augmentation severity is asymmetric.
- *VAD-empty post-processing* (§7.3) — eval-time correctness fix for inputs the feature extractor maps to a degenerate zero vector. Does not affect clean dev.
- *Stress dev construction* (§4.2) — diagnostic, not a model change.

These appear in the dokumentation under their own sections rather than as steps in the progression, because they don't make the AUC number on the held-out clean split go up.
