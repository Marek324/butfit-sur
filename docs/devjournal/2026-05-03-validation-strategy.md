# Validation strategy — 2026-05-03

## Holdout dev (current default)

Used the original `target_train` / `target_dev` split as holdout. target_train is m431 session 01 (20 paired samples), target_dev is session 03 (10 paired samples). Non-target sets have disjoint impostors between train and dev — that part is good, it pre-stresses generalization to unseen impostors.

## Session-grouped CV scaffolded but not used

`src/cv.py` implements `session_grouped_folds` over `<speaker>_<session>` filename groups (instructor's recommendation — no session straddles a fold boundary). It's tested but not wired into the orchestrator. Reason: with only 2 distinct m431 sessions across train+dev (01 and 03), K-fold doesn't give much beyond what holdout already provides — each fold would just swap which session is in train/dev, and we'd average two near-symmetric measurements. Not free signal.

## Clean dev is too easy — stress-dev reveals generalization gap

After offline augmentation + class-symmetric noise: image and fusion both saturate at AUC = 1.0 on clean dev. ROC/DET curves collapse onto the corner — uninformative.

target_train (session 01) and target_dev (session 03) are different recording sessions but the underlying face/voice are similar enough that a 2916-d HOG descriptor trivially separates m431 from any of 60 unseen impostors. The split was working as intended for the impostor-generalization question; it's just too easy for the question of *"does the model overfit to session-specific cues that won't transfer to eval?"*.

To probe that, built a stress dev: same dev samples, augmented with a *disjoint* envelope (`configs/augment_dev_*.toml`). Disjoint means the augmentation ranges have no overlap with the train envelope, so anything the model learned as "invariant" during training is now extrapolation rather than interpolation.

Stress-dev metrics:

| | clean dev AUC | stress dev AUC | clean EER | stress EER |
|---|---|---|---|---|
| audio  | 0.972 | **0.529** | 4.17%  | **46.67%** |
| image  | 1.000 | **0.603** | 0.00%  | **40.42%** |
| fusion | 1.000 | **0.623** | 0.00%  | **39.58%** |

Audio collapses to near-chance under stress — it's brittle to gain attenuation + heavy noise + EQ shifts. Image holds up partially (60% AUC). Fusion is modestly better than image alone — that 0.62 > 0.60 is the structural confirmation that fusion helps, which we couldn't see in the saturated clean plot.

Caveat: my stress envelope is *guessed*. The assignment says "noise, changes in image quality" without specifying severity — I picked something deliberately harsh (gain ∈ [-15, -8] dB, SNR ∈ [3, 13] dB, brightness ∈ [0.45, 0.70], etc.). If actual eval is gentler, these numbers are pessimistic. But the *gap* between clean and stress is real, and that's the diagnostic.

## Lesson

Clean-dev metrics on this split are mostly a "no obvious bug" check. They don't predict eval. Stress dev is the more honest generalization estimate, and it should be the metric I track when iterating on features and classifiers.
