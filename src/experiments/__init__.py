"""End-to-end experiment runner.

Public surface:
    run_experiment(cfg, out_dir) -> dict
        Train + evaluate the system described by `cfg`. Writes
        artifacts/<out_dir>/{model.pkl, train_features.npz, metrics.json}.

    extract_set(...) — shared per-modality feature extraction with per-sample
                      caching (audio uses a 2-stage cache: heavy pre-stage +
                      cheap aggregation, see extraction.py).

    score_unlabeled(...) — score a list of samples through a trained model.

    compare_runs(...) — side-by-side text table from metrics.json sidecars.

Side-by-side variants are made cheap by:
  - hashing the feature config so re-runs that change classifier-only reuse
    cached features (this also caches the offline-generated augmented samples),
  - 2-stage audio caching: variants that change only `include` / `mfcc_n_means`
    re-aggregate from cached pyin/MFCC/formant intermediates (~10 ms vs ~2 s),
  - writing a JSON metrics sidecar so a later "compare runs" script can read
    multiple artifact dirs without re-running anything.
"""

from .compare import compare_runs
from .extraction import (
    extract_one_audio,
    extract_one_image,
    extract_set,
    feature_config_hash,
)
from .runners import run_audio_or_image, run_experiment, run_fusion, score_unlabeled

__all__ = [
    "compare_runs",
    "extract_one_audio",
    "extract_one_image",
    "extract_set",
    "feature_config_hash",
    "run_audio_or_image",
    "run_experiment",
    "run_fusion",
    "score_unlabeled",
]
