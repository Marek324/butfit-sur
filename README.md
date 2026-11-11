# butfit-sur

butfit-sur is our project for the SUR course (Machine Learning and Recognition) at FIT BUT, 2025/2026.
It detects one target person (`m431`) from a face image (PNG) and a voice recording (WAV).

It trains image-only, audio-only and fusion systems.
Each one writes a score and a hard decision for every eval segment to `results/`.
The assignment is in `docs/assignement/`, and the full write-up with results is in [`docs/dokumentace.md`](docs/dokumentace.md).

## How it works

- **Augmentation:** each training sample gets 5–15 variants: noise, gain, time stretch and EQ for audio; flips, rotation, shifts, brightness and noise for images.
- **Image:** HOG features from 80×80 grayscale images, classified by balanced logistic regression.
- **Audio:** energy/ZCR voice activity detection (VAD), then MFCC and prosodic features. Two systems use them: logistic regression on per-sample statistics, and a GMM-UBM (16-mixture background, 4-mixture target) scored by log-likelihood ratio. In the hybrid version, a GMM trained without VAD takes over for clips where VAD trims every frame.
- **Fusion:** logistic regression over the audio and image scores, trained on the dev split.
- **Validation:** a clean dev split, plus an augmented "stress" dev split to check robustness.

| Result file | System |
|---|---|
| `image_logreg.txt` | image, HOG + logistic regression |
| `audio_logreg.txt` | audio, logistic regression |
| `audio_gmm_hybrid.txt` | audio, GMM-UBM with no-VAD fallback |
| `fusion_logreg.txt` | fusion of `audio_logreg` + `image_logreg` |
| `fusion_hybrid_image.txt` | fusion of `audio_gmm_hybrid` + `image_logreg` |
| `fusion_3way.txt` | fusion of both audio systems + `image_logreg` |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.
`scripts/submit.py` also needs Pango, which WeasyPrint uses to render the PDF.

```bash
uv sync
```

Download [SUR_projekt2025-2026.zip](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2025-2026/SUR_projekt2025-2026.zip) and unzip its `target_*` and `non_target_*` folders into `dataset/`.
Then fetch the eval set:

```bash
uv run scripts/download_dataset.py
```

## Reproduction

Run these from the repo root, in this order:

```bash
uv run scripts/augment_dataset.py --config configs/augment_audio.toml
uv run scripts/augment_dataset.py --config configs/augment_image.toml
uv run scripts/train.py --config configs/audio.toml
uv run scripts/train.py --config configs/image.toml
uv run scripts/train.py --config configs/fusion.toml
uv run scripts/evaluate.py --config configs/audio.toml --split eval
uv run scripts/evaluate.py --config configs/image.toml --split eval
uv run scripts/evaluate.py --config configs/fusion.toml --split eval
uv run scripts/augment_dataset.py --config configs/augment_dev_audio.toml
uv run scripts/augment_dataset.py --config configs/augment_dev_image.toml
uv run scripts/evaluate.py --config configs/audio_stress.toml --split dev
uv run scripts/audio_gmm_pipeline.py --mode map_free
uv run scripts/audio_gmm_hybrid.py
uv run scripts/fix_vad_empty.py
uv run scripts/fusion_hybrid_image.py
uv run scripts/fusion_3way.py
```
