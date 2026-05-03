# 50/50 prior enforcement — 2026-05-03

The assignment specifies: *"Hard decision: 1 if score ≥ threshold else 0, assuming a prior probability of 0.5."* This has to be enforced consistently across audio, image, and fusion. Two parts:

## 1. Train as if classes are balanced

Every classifier (audio logreg, image logreg, fusion logreg, optional SVMs) uses `class_weight='balanced'`. sklearn weights each class's loss inversely to its frequency in train; effectively the optimizer sees a balanced dataset. The trained logit's intercept then corresponds to *P(target=1|x) = 0.5 at logit = 0*, regardless of actual training imbalance.

After offline augmentation: **320 target / 792 non-target.** Without `class_weight='balanced'`, the logreg intercept would absorb the target-minority bias and the threshold-0 decision would skew toward predicting non-target. With `balanced`, it doesn't:
- target weight = (320 + 792) / (2 × 320) = 1.74
- non-target weight = (320 + 792) / (2 × 792) = 0.70
- weighted mass per class = 320 × 1.74 = 792 × 0.70 = 556 ✓

Equal effective mass per class.

For SVMs, `class_weight='balanced'` shifts the hyperplane the same way — `decision_function = 0` corresponds to balanced classes regardless of training counts.

## 2. Threshold the score at 0

`output.hard_threshold_logit = 0.0` in every config (`audio_baseline.toml`, `image_baseline.toml`, `fusion_baseline.toml`). For logreg, that's the P(target) = 0.5 boundary; for SVM, it's the trained decision boundary; for fusion logreg, same logic.

The score column in the result file is a real number, monotonic with target probability, sign-aligned to "1 = target". Higher = more m431. Hard decision is `score ≥ 0`.

## What broke and what didn't

After noise was added to audio augmentation, the system started misbehaving at threshold-0 (TP=4/10 instead of 8/10) even though `class_weight='balanced'` was set. Initially looked like a prior-enforcement bug; turned out to be the class-asymmetric noise issue (see augmentation entry).

**Distinction matters:** the 50/50 prior is a *decision rule*, not a guarantee about the score distribution. With `class_weight='balanced'` + threshold = 0, you implement the correct decision rule. Whether the rule produces the *Bayes-optimal* error rate depends on whether the trained model approximates the true posterior — which it can fail to do for unrelated reasons (overfitting, train/test mismatch, asymmetric augmentation).

So the 50/50 prior is correctly enforced everywhere ✓. But the rule's actual error rate at threshold-0 depends on the model's calibration, not on the rule itself.

## Where it's documented in code

Comments in the relevant config blocks say so explicitly:
- `audio_baseline.toml`, `image_baseline.toml`: `class_weight = "balanced"  # enforces 50/50 prior` and `hard_threshold_logit = 0.0  # logit=0 ⇔ P(target)=0.5`
- `fusion_baseline.toml`: same, plus a note that fusion balances against the dev-split ~10/60 ratio

The classifier module's docstring also explains that both `LogisticRegression.decision_function` and `SVC.decision_function` are sign-aligned to class label 1, so threshold = 0 means the same thing across classifier types.
