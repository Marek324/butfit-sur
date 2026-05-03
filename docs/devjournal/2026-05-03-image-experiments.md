# Image feature/classifier experiments — 2026-05-03

Ran 6 image variants comparing on **stress dev** (configs/augment_dev_image.toml). Clean dev saturates at AUC=1.0 for almost all of them, so it's not diagnostic — only stress dev separates them.

## Results

| variant | clean AUC | stress AUC | stress EER | feature dim | classifier |
|---|---|---|---|---|---|
| baseline | 1.000 | 0.603 | 40.42% | 2916 | logreg C=1.0 |
| hog_color (RGB) | 0.997 | 0.597 | 46.75% | 2916 | logreg C=1.0 |
| hog_coarser (16×16 cells) | 0.987 | **0.616** | 40.83% | 576 | logreg C=1.0 |
| hog_finer (4×4 cells) | 0.998 | 0.575 | 41.83% | 12996 | logreg C=1.0 |
| **strong_reg** | **1.000** | 0.607 | **40.00%** | 2916 | logreg C=0.1 |
| svm_rbf | 1.000 | 0.601 | 45.92% | 2916 | svm_rbf |
| combo (coarser + C=0.1) | 0.963 | **0.625** | 40.00% | 576 | logreg C=0.1 |

## Reading

- **HOG color is worse** (47% EER vs 40% baseline): RGB doesn't add identity signal beyond gradient orientation; it does add channels of noise the model can latch onto.
- **Finer HOG is worse** (12996-d, EER=42%): the overfitting-control hypothesis confirmed — more capacity → worse stress generalization.
- **Coarser HOG is slightly better on AUC** (0.616) but slightly worse on EER and *much* worse on clean dev (0.987 → loses 1.3% AUC). Trade-off.
- **Strong regularization (C=0.1)** maintains clean dev = 1.0 AND has the best stress EER (40.00%). Same feature dim as baseline so cache stays valid.
- **SVM-RBF** doesn't help — non-linear kernel doesn't find structure linear didn't, and decision_function semantics are messier for fusion.
- **Combo (coarser + C=0.1)** wins on stress AUC (0.625) but takes a 4-point hit on clean dev. The capacity is now too low for the easy task to be solved perfectly.

## Decision: promote `strong_reg`

Update `image_baseline.toml` to `C = 0.1`. Reasons:
- Best stress EER (40.00% — tied with combo)
- Maintains clean-dev AUC=1.0 — no regression on the easy split
- Same feature dim as current baseline → existing feature cache reused, no fresh extraction
- Trivial config change

The combo (coarser + C=0.1) is the better stress AUC but we accept a 4% clean-dev hit for a 1.8% stress AUC gain. Strong-reg-only is the more conservative pick — it doesn't sacrifice clean-dev performance to gain marginal stress AUC.

## Caveat

All these stress AUCs are in the 0.57–0.63 band — the spread is small (6 points). With 70 dev samples × 5 augmentations = 350 stress samples, the standard error on AUC is ~3–4 points. So the differences between top variants are within noise. The ordering is suggestive, not statistically firm.

The eval set is the real test. Tomorrow's eval results may rank these differently.
