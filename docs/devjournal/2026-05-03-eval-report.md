# Eval-set scoring report - 2026-05-03

> **Status: in progress.** Image scored, audio + fusion pending audio extraction.

The eval set is `dataset/eval/` - 1346 paired samples, unzipped from `SUR_projekt2025-2026_eval.zip`. Filenames are `eval_<NNNN>.{wav,png}`. No labels (assignment design - answer key publishes 2026-05-05).

Pipeline:
- `scripts/evaluate.py --config <baseline> --split eval` → writes `results/<modality>_logreg.txt` in submission format (`<segment_name> <score> <hard_decision>`).
- Hard decision uses `hard_threshold_logit = 0.0` (50/50-prior decision rule).

## Image (production: HOG 8×8, logreg C=0.1)

```
n=1346, predicted_target=127 (9.4%)
score: min=-8.77, q25=-4.86, med=-3.41, q75=-1.90, max=+4.71
```

The score distribution is heavily negative-skewed: median is -3.41, only 9.4% of segments score above 0. This is consistent with a small minority of the eval set actually being m431 - most face crops are impostors and the model rates them strongly negative.

The **127 predicted target** number is what the 50/50-prior decision rule produces, regardless of actual eval prevalence. If the answer key shows the true target prevalence is much lower or higher than 9.4%, we'll be over- or under-predicting - but the rule is fixed by the assignment.

## Audio + fusion: pending

Audio extraction over 1346 `.wav` samples through `librosa.pyin` is ~30 min single-threaded - the eval scoring is intentionally not run yet because:

1. Audio feature experiments are still running (deciding the production audio config). If a non-baseline variant wins, eval features for that config still need extracting; doing it now and again later doubles the cost.
2. Fusion eval depends on audio model `decision_function` over eval, so it inherits the same wait.

Plan: complete audio experiments → promote audio winner → score audio + fusion eval in one final extraction pass.

## How to interpret submission scores

The score column is `decision_function` output:
- For logreg: log-odds of P(target=1|x). Higher is more m431.
- For fusion logreg-on-stacked-scores: same, after the fusion logreg's own affine combination.

Sign convention is consistent across all three result files (positive = target). Hard decision is `score >= 0`.
