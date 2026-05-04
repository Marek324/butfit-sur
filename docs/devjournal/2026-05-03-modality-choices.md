# Modality choices - 2026-05-03

What we landed on:
- **Audio**: 10-d hand-crafted utterance descriptor (5 prosodic + 5 CMVN-MFCC means) → balanced logistic regression
- **Image**: 80×80 grayscale → HOG (9 orientations, 8×8 cells, 2×2 cell blocks → 2916-d) → balanced logistic regression
- **Fusion**: score-level - 2-input balanced logreg over per-modality `decision_function` outputs

## Why not GMM-UBM for audio (yet)

The syllabus default is MFCC + GMM-UBM with MAP-adapted target model. With ~30 target utterances after augmentation and ~150 non-target, fitting both a UBM and a target model is borderline - especially without K-fold averaging. The compact 10-d descriptor lives in a much smaller space (matched to data size) and balanced logreg gives calibrated log-odds that play nicely with score-level fusion downstream. GMM-UBM is reserved as a fallback if logreg underperforms in eval.

## Why not a from-scratch CNN for image

1112 training samples is far below what a CNN needs to learn 80×80 face features from scratch. No pretrained models allowed (assignment constraint). HOG is the strongest hand-crafted alternative - it captures local edge-orientation distributions in a way that's proven for face/pedestrian tasks, and it has zero learned parameters. SVM-RBF on HOG was considered; logreg was picked because its `decision_function` is calibrated log-odds, keeping the 50/50-prior threshold = 0 invariant clean across modalities.

## Why score-level fusion (not feature-level)

Feature-level fusion would mean stacking a 10-d audio vector with a 2916-d image vector → ~2926-d joint feature against ~1100 training samples. Curse of dimensionality, no obvious cross-modality normalization, and impossible to test each modality independently.

Score-level fusion: each modality keeps its own pipeline and produces a real-valued score; a 2-feature logreg fits the optimal weighting and absorbs whatever scale each upstream classifier produces. Almost impossible to overfit a 2-feature model. It's also what production biometric-verification systems actually do, for the same reasons.

Caveat: the fusion classifier is currently trained on the dev split, so reported fusion dev metrics are biased optimistic. CV-OOF scores from per-modality models would give an unbiased fusion training set, but it's not wired up.
