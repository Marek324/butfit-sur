# Experiment scaffold - 2026-05-03

The project is experiment-driven by necessity (small dataset, many design choices). The scaffold makes side-by-side variants cheap.

## Per-sample feature cache

`src/cache.py`. Layout: `artifacts/cache/<modality>/<config_hash>/<stem>.npy`. The config hash is derived from the relevant feature config (`AudioFeaturesCfg + VadCfg` for audio, `ImageFeaturesCfg` for image), so re-running with the same features but a different classifier reuses extraction. Re-running with different features writes a new cache directory - old features stay valid for the original config.

The cache is mtime-aware: if the source file is newer than the cached entry, the entry is invalidated. This was added after re-augmenting silently kept stale features (the augmented filenames don't change, only the bytes).

## Metrics sidecars

Each training run writes `artifacts/<name>/metrics.json` with AUC, EER, confusion, and metadata. `scripts/run_experiments.py` reads these sidecars from multiple artifact dirs and prints a comparison table. No re-training needed for the comparison - just the JSON files.

## Modality-dispatching orchestrator

`src/experiments.run_experiment(cfg, out_dir)` switches on `cfg.modality`:
- audio/image: extract features (with cache), fit, evaluate on dev, write artifacts
- fusion: load two pre-trained per-modality models, score paired dev samples, fit fusion logreg, write artifacts

`scripts/train.py` and `scripts/evaluate.py` both delegate to this orchestrator, so adding a new modality (e.g., a CNN on raw pixels) is one new branch in one function rather than parallel scripts.

## Strict TOML config validation

`src/config.py` parses TOML into frozen dataclasses with strict unknown-key rejection. Typo in a key raises immediately rather than silently using defaults. The dataclass tree mirrors the TOML hierarchy 1:1 with one piece of "flattening" sugar (the old `[audio.augment.train]` key - now removed since augmentation moved offline).

## ROC + DET plotting

`scripts/plot_curves.py` takes 1+ configs, scores them on dev (or any directory pair via `--dev-dirs`), and emits a 2-panel ROC + DET PNG. EER points marked, AUC in legend. DET uses probit-warped axes (the standard biometric-verification convention; well-behaved systems plot as nearly straight lines). When EER = 0 the marker clamps to the bottom-left corner of the plot so perfect classifiers stay visible.

## What this enables

- Re-running an experiment is cheap when features dominate cost (audio MFCC extraction is the bottleneck, ~1.5 s/sample uncached, ~0 cached).
- Comparing multiple configs is just `run_experiments.py cfg1.toml cfg2.toml ...` - sidecars give the table.
- `plot_curves.py --dev-dirs` lets us swap the evaluation set without touching baseline configs (used heavily once stress-dev was introduced).

## Trade-offs the scaffold is *not* paying for

- No multiprocessing in extraction. Audio extraction is single-threaded (librosa.pyin). 1346 eval samples take ~30 min. A worker pool would cut that significantly but adds complexity for limited benefit on this dataset size.
- No CV-OOF support. Folds are scaffolded in `src/cv.py` but not wired into the orchestrator. With only 2 m431 sessions, K-fold doesn't add much.
- No automatic hyperparameter sweep. Each variant is its own TOML; sweeps are manual lists of configs to `run_experiments.py`.

These are intentional - the scaffold optimizes for cheap iteration on *which features* and *which classifier*, not for hyperparameter optimization.
