# Validation strategy - 2026-05-03

## Holdout dev (current default)

Used the original `target_train` / `target_dev` split as holdout. target_train is m431 sessions 01 and 02 (20 paired samples, 10 prompts × 2 sessions), target_dev is session 03 (10 paired samples). Non-target sets have disjoint impostors between train and dev - that part is good, it pre-stresses generalization to unseen impostors.

## Session-grouped CV scaffolded but not used

`src/cv.py` implements `session_grouped_folds` over `<speaker>_<session>` filename groups (instructor's recommendation - no session straddles a fold boundary). It's tested but not wired into the orchestrator.

The original justification in this entry undercounted the m431 sessions (claimed 2 across train+dev, actually 3: train holds 01 and 02, dev holds 03). With three sessions a leave-one-session-out 3-fold would have produced strictly more signal than the holdout, and the "two near-symmetric measurements" argument doesn't apply. Not run for the submission - flagging the miscount honestly rather than rewriting the decision after the fact.

## Clean dev is too easy - stress-dev reveals generalization gap

After offline augmentation + class-symmetric noise: image and fusion both saturate at AUC = 1.0 on clean dev. ROC/DET curves collapse onto the corner - uninformative.

target_train (sessions 01 + 02) and target_dev (session 03) are different recording sessions but the underlying face/voice are similar enough that a 2916-d HOG descriptor trivially separates m431 from any of 60 unseen impostors. The split was working as intended for the impostor-generalization question; it's just too easy for the question of *"does the model overfit to session-specific cues that won't transfer to eval?"*.

To probe that, built a stress dev: same dev samples, augmented with a *disjoint* envelope (`configs/augment_dev_*.toml`). Disjoint means the augmentation ranges have no overlap with the train envelope, so anything the model learned as "invariant" during training is now extrapolation rather than interpolation.

Stress-dev metrics:

| | clean dev AUC | stress dev AUC | clean EER | stress EER |
|---|---|---|---|---|
| audio  | 0.972 | **0.529** | 4.17%  | **46.67%** |
| image  | 1.000 | **0.603** | 0.00%  | **40.42%** |
| fusion | 1.000 | **0.623** | 0.00%  | **39.58%** |

Audio collapses to near-chance under stress - it's brittle to gain attenuation + heavy noise + EQ shifts. Image holds up partially (60% AUC). Fusion is modestly better than image alone - that 0.62 > 0.60 is the structural confirmation that fusion helps, which we couldn't see in the saturated clean plot.

Caveat: my stress envelope is *guessed*. The assignment says "noise, changes in image quality" without specifying severity - I picked something deliberately harsh (gain ∈ [-15, -8] dB, SNR ∈ [3, 13] dB, brightness ∈ [0.45, 0.70], etc.). If actual eval is gentler, these numbers are pessimistic. But the *gap* between clean and stress is real, and that's the diagnostic.

## Lesson

Clean-dev metrics on this split are mostly a "no obvious bug" check. They don't predict eval. Stress dev is the more honest generalization estimate, and it should be the metric I track when iterating on features and classifiers.

## 2026-05-04 follow-up: session-grouped CV diagnostic

Ran 2-fold leave-one-target-session-out CV on the training pool (`scripts/cv_session_check.py`, output in `artifacts/cv_session_check.txt`). Same logreg config as production; non-target groups split by session (so a non-target speaker can have one session in train and another in CV-dev — session-disjoint but not speaker-disjoint).

| Modality | Holdout clean dev AUC | CV mean AUC |
|---|---:|---:|
| Audio | 0.983 | **0.789** (0.794, 0.785) |
| Image | 1.000 | 0.982 (0.976, 0.988) |

Caveat first: each CV fold trains on only one m431 session (half the production target data), so part of the audio drop is "less training data" and part is "session-shift sensitivity". The CV result is **not** a strictly fairer estimate of the holdout — it's a different test.

What it does separate cleanly:
- Image features are session-invariant. The 1.0 holdout AUC is genuine (HOG over centred faces is identity-geometric, not session-dependent).
- Audio is session-sensitive. Two-session training is doing real work for the holdout 0.98; one-session training collapses to 0.79. This validates the design choice to train on the full `target_train` (sessions 01 + 02) and motivates the disjoint-envelope stress dev (§Stress dev) — clean dev alone cannot tell "model generalises" from "model interpolates between two seen sessions to a third".

Defensible oral-defence narrative: holdout dev measures speaker-disjoint generalisation given full training; session-grouped CV measures session-shift sensitivity given reduced training; stress dev measures distribution-shift sensitivity. The three together explain why audio is the modality at risk on eval and why image carries the fusion.
