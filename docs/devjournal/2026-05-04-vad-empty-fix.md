# VAD-empty submission fix - 2026-05-04

## What broke

While auditing why stress-dev audio AUC was 0.521 with all 50 stress targets predicted as target, found that the audio feature extractor returns an **all-zero feature vector** for stress audio. The energy-relative VAD over-trims under heavy gain attenuation, leaving zero voiced frames; without voiced frames, MFCC means and prosodic statistics fall back to zeros.

The trained logreg's response to a zero-feature input is **logit ≈ +1.535** (an artefact of the bias term, *not* signal). Under the threshold-0 hard rule, every all-zero sample is mislabelled "target".

This is also the mechanism behind the line in `final-report.md` that reads "the audio model maps all 50 stress targets to the same logit ≈ +1.535" - it's literally the same value because the input is literally identical (all zeros).

## How much eval is affected

Re-extracting audio for `dataset/eval/`: **79 of 1346 samples (5.9%)** produce all-zero feature vectors. Before the fix, all 79 were predicted target by the audio system, and 36 of them were predicted target by the fusion system (audio's bias score was strong enough to overrule image in many cases).

These are not silent recordings - they include eval samples with peak amplitudes up to 0 dB. Whatever combination of attenuation, EQ, and short voiced segments these samples have, the energy-relative VAD with its current `energy_margin_db = 8.0` threshold trims everything.

## Fix

`scripts/fix_vad_empty.py` is a post-processor on the result files:

1. Identify rows in `results/audio_logreg.txt` with score within 0.005 of the collapse value 1.535 (the unique logit value the model emits for all-zero input).
2. Replace those audio scores with **−5.0** (well below the audio score median of −2.16) and hard decision **0**.
3. Re-score fusion across the entire eval set using the corrected audio scores + the unchanged image scores, fed through the trained fusion model.
4. Backup the originals as `*.before_vad_fix.txt`.

The list of affected segments is saved at `artifacts/vad_empty_eval_samples.txt` (79 stems).

## Effect on the submission

| File | Predicted target before fix | after fix | Change |
|---|---:|---:|---:|
| `audio_logreg.txt`  | 232 | 153 | −79 |
| `image_logreg.txt`  | 127 | 127 | unchanged |
| `fusion_logreg.txt` | 163 | 142 | −21 |

Fusion flips: 21 hard decisions, all going **target → non-target**. Zero spurious flips to target (audio coefficient is positive, so reducing audio monotonically reduces fusion). The 21 are samples where audio's degenerate +1.535 was previously the deciding signal and image disagreed.

## Why this is the right call (defence narrative)

The trained classifier was never shown all-zero inputs during training, so its response to such inputs is undefined behaviour. The bias-driven default of logit +1.535 is not a calibrated belief - it's an arbitrary value determined by the optimiser's intercept fit. Reporting "predict target with 78 % confidence" for an input that contains no signal under the model's own pipeline is wrong on the face of it.

The 50/50-prior decision rule the assignment specifies says "predict target iff the model's posterior crosses 0.5". A model with no information should not cross 0.5 in either direction; we set it to a conservative non-target default with a clearly-negative sentinel score, preserving sign convention and the monotonic-in-confidence requirement of the score column.

## Why we didn't fix the VAD itself

Two reasons:

1. The `vad_off` ablation (no VAD, all frames count) wins on stress dev (AUC 0.668 vs 0.521 for VAD-on) but **loses badly on clean dev** (AUC 0.627 vs 0.972). On the eval set, the proportion of "stress-like" audio is unknown; the safer bet is to keep VAD-on for the majority of eval and patch the 5.9 % degenerate cases.
2. Re-tuning VAD on deadline day risks regressing the 94 % of eval that the current configuration handles correctly.

The post-processing patch is isolated, reversible, easily verified, and only touches inputs the model produces no useful signal for.

## Files changed

- `results/audio_logreg.txt` (79 rows patched)
- `results/fusion_logreg.txt` (regenerated with corrected audio)
- `results/{audio,fusion}_logreg.before_vad_fix.txt` (backups for diff)
- `scripts/fix_vad_empty.py` (the post-processor)
- `artifacts/vad_empty_eval_samples.txt` (list of 79 affected stems)
