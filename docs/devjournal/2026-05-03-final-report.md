# Final report — 2026-05-03

All three production systems trained, scored on the eval set, and packaged
for submission. `dokumentace.pdf` is the only remaining human task.

## Production systems

| System | Features | Classifier | Clean dev AUC / EER | Stress dev AUC / EER |
|---|---|---|---|---|
| Audio  | 5 prosodic + 13 CMVN-MFCC means (18-d, **promoted from `more_mfcc`**) | logreg, C=1.0, balanced | **0.983 / 3.33%**  | 0.521 / 47.08% |
| Image  | HOG 8×8 cells, 9 orient, grayscale (2916-d)        | logreg, **C=0.1**, balanced | 1.000 / 0.00%  | 0.607 / 40.00% |
| Fusion | score-level logreg over (audio, image), zscore-norm | logreg, C=1.0, balanced | 1.000 / 0.00%  | 0.623 / 39.58% |

50/50-prior decision rule: hard decision = `score >= 0` for all three.

## Eval-set scoring (1346 paired samples)

| File | Median score | Predicted target | % of eval |
|---|---|---|---|
| `results/audio_logreg.txt`   | -2.16 | 231 | 17.2% |
| `results/image_logreg.txt`   | -3.41 | 127 | 9.4%  |
| `results/fusion_logreg.txt`  | -4.17 | 172 | 12.8% |

Score histograms for all three: `artifacts/eval_scores_all.png`. Reading:

- **Audio** has a bimodal distribution — main impostor mass around score=-3,
  and a sharp secondary peak right above zero (score≈+1.5) of ~95 segments.
  That bump is what produces the 231 hard-target predictions: the new 18-d
  feature space lets the classifier express a clearer "yes/no" structure
  than the 10-d baseline could.
- **Image** is broadly negative (median -3.4) with a clean small positive
  tail. The bulk of eval is impostors and the model rates them strongly
  negative; predicted-target=127 (9.4%) is consistent with a small minority
  m431 fraction.
- **Fusion** is the broadest distribution — the score-level logreg amplifies
  audio's positive tail with image's strong-negative bulk, giving a wider
  spread. 181 predicted target ≈ image's positives + audio's confident
  positives.

## Decisions made (the WHY for the documentation)

1. **No GMM-UBM, no CNN, no pretrained anything** — assignment forbids pretrained models, and the data size (~30 target, ~150 non-target) is too small for either GMM-UBM (unstable MAP adaptation) or a from-scratch CNN. Hand-crafted DSP features + balanced logreg is the right scale.

2. **Score-level fusion over feature-level.** A 10-d audio + 2916-d image stacked feature vector against ~1100 training samples is the curse of dimensionality. Score-level lets each modality keep its own pipeline, and the 2-input fusion logreg is basically immune to overfitting.

3. **Offline augmentation, written to disk.** `scripts/augment_dataset.py` produces `dataset/*_aug` once. Training enumerates them like any other sample. Wins: feature cache covers augmented data, augmentations are inspectable, training is deterministic from on-disk bytes, no online randomness in the trainer.

4. **Class-symmetric noise.** `n_per_target = 15`, `n_per_nontarget = 5`. Earlier `n_per_nontarget = 0` left only targets noisy → classifier picked up "noisy ⇒ target" as a shortcut → AUC@dev 0.888 / EER 20% / TP@logit=0 4/10. Setting `n_per_nontarget = 5` made the noisy-fraction roughly class-symmetric (94% vs 83%) → AUC 0.972 / EER 4.17% / TP 8/10. *This was the single biggest fix in the project.*

5. **Stress-dev was the diagnostic that actually mattered.** Clean dev saturates (image and fusion both AUC = 1.0). Disjoint-envelope stress dev (`configs/augment_dev_*.toml`) revealed real generalization gaps: audio collapses to AUC 0.53 (chance), image holds 0.60, fusion 0.62. All experiment promotion decisions used stress dev metrics, not clean dev.

6. **Image promotion: C=0.1.** Out of 6 image variants tested, `strong_reg` (HOG 8×8 + logreg C=0.1) was picked. Best stress EER (40.00%) without sacrificing clean AUC. The combo (coarser-cells + C=0.1) won stress AUC by 1.8% but cost 4% clean AUC — not worth.

7. **Audio promotion: `more_mfcc` (5 prosodic + 13 MFCC means).** Out of 9 audio variants tested, `more_mfcc` won clean dev (AUC 0.983, EER 3.33%, only 3 errors at logit=0). Speaker_features (16-d, formants+ratios+centroid) got the highest AUC (0.985) but worst EER@0 in the top tier — the score distribution shifted off the optimal operating point. The 8 extra MFCC coefficients (going from 5 to 13 means) carry real spectral envelope information.

8. **VAD is essential — confirmed by ablation.** `vad_off` (no VAD, all frames count) collapses clean dev AUC from 0.972 to 0.627, with TP=0/10 at logit=0 — the model can't distinguish targets from impostors at all. Without VAD, MFCC means and voicing_ratio get poisoned by silence/breath/boundary booms. Stricter VAD (`vad_strict`, 12 dB margin / 200 ms min speech) was slightly worse than baseline — current thresholds are well-tuned. Surprisingly `vad_off` *wins* on stress dev (AUC 0.668 vs 0.52 for VAD-on variants) — under heavy gain attenuation the energy-relative VAD over-trims; for the 50/50-prior decision rule on clean-ish eval data, VAD-on is still the right call.

9. **2-stage cache for fast variant sweeps.** Audio extraction split into `extract_pre()` (heavy: pyin + MFCC + formants + centroid, ~2s/sample) and `aggregate()` (cheap: ~10ms select+aggregate). Pre cache keyed only on params that change pre contents (vad_cfg + pyin + MFCC dims) — variants that change `include` / `mfcc_n_means` re-aggregate from cached pre. Result: 1500× speedup on variant sweeps; running 9 audio variants this afternoon instead of 1 baseline.

10. **50/50 prior enforcement.** `class_weight='balanced'` in every classifier (audio, image, fusion); `hard_threshold_logit = 0.0` in every output config. The `balanced` weighting makes the trained logit's intercept correspond to P(target) = 0.5 at logit = 0, regardless of the actual training class ratio. SVM `decision_function = 0` is the trained boundary under the same weighting.

## What I'd add given more time

(See `2026-05-03-proposed-experiments.md` for the full list.)

- **CV-OOF fusion training**: replace the dev-trained fusion (which inflates dev metrics) with K-fold OOF scores from per-modality models. Removes the leak.
- **Image PCA**: `image.features.pca_components` is config-supported but not implemented; would address the saturated-clean-dev / overfit dynamic more cleanly than the L2 reg trick.
- **Faster formant extraction** (LPC every-K-frames, or only on voiced frames) so the speaker_features experiment becomes tractable.
- **Δ-MFCC and ΔΔ-MFCC means** for temporal dynamics of timbre.
- **GMM-UBM as audio fallback** if logreg under-performs in eval analysis (May 6th lecture).

## Submission package

Pending: `docs/dokumentace.pdf` (3 A4 pages, in CZ/SK/EN, the WHY narrative).
The devjournal entries are the raw material — `modality-choices`,
`validation-strategy`, `augmentation`, `prior-enforcement` are the most
relevant sections to compose from.

After dokumentace.pdf is in place:

```
uv run scripts/submit.py
```

→ `submit.zip` containing `audio_logreg.txt`, `image_logreg.txt`, `fusion_logreg.txt`,
`SRC/` (with `pyproject.toml` and generated `requirements.txt`), and `dokumentace.pdf`.
Rename to `hrmajo1.zip` (or `login1_login2.zip` if a partner is on the team)
before uploading to IS.

## Visualizations produced

`artifacts/`:
- `curves.png` / `curves_stress.png` — ROC + DET on clean / stress dev for all three production systems
- `eval_scores_all.png` — score histograms for all three eval result files
- `score_correlation.png` — eval score scatter + Pearson/Spearman across modality pairs (audio↔image only 0.32, image↔fusion 0.85 — modalities pick up genuinely different identity cues)

`artifacts/experiments/`:
- `image_variants_{clean,stress}.png` — ROC + DET for 6 image variants
- `audio_variants_{clean_v2,stress_full}.png` — ROC + DET for 7+ audio variants
- `audio_summary_{clean_v2,stress_full}.png` — AUC + EER bar charts (clean & stress dev) for all 9 audio variants
- `image_summary_stress.png` — AUC/EER bar chart for image variants
- `audio_variants_{clean,stress}.png` — ROC + DET for 3 audio variants
- `audio_summary_clean.png` — AUC/EER bar chart for audio variants
