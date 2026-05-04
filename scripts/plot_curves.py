#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from classifier import load as load_classifier  # noqa: E402
from config import Config, load_config  # noqa: E402
from data import Sample, enumerate_paired, enumerate_split  # noqa: E402
from experiments import extract_set  # noqa: E402
from fusion import load as load_fusion, score_fusion  # noqa: E402

DET_TICKS = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4)
DET_AXIS_RANGE = (5e-5, 0.5)


def _load_artifact_config(d: Path) -> Config:
    return load_config(d / "config.toml")


def _score_unimodal(
    cfg: Config, artifacts_root: Path, modality: str, dev_dirs: list[Path]
) -> tuple[np.ndarray, np.ndarray]:
    ext = ".wav" if modality == "audio" else ".png"
    samples = enumerate_split(dev_dirs, extension=ext)
    model = load_classifier(artifacts_root / cfg.name / "model.pkl")
    X, _y, _names, _ = extract_set(samples, cfg, modality)
    scores = np.asarray(model.estimator.decision_function(X), dtype=np.float64)
    y = np.asarray([s.label for s in samples])
    return scores, y


def _score_fusion(
    cfg: Config, artifacts_root: Path, dev_dirs: list[Path]
) -> tuple[np.ndarray, np.ndarray]:
    fmodel = load_fusion(artifacts_root / cfg.name / "model.pkl")
    audio_dir = artifacts_root / cfg.fusion.audio_artifact
    image_dir = artifacts_root / cfg.fusion.image_artifact
    audio_model = load_classifier(audio_dir / "model.pkl")
    image_model = load_classifier(image_dir / "model.pkl")
    audio_cfg = _load_artifact_config(audio_dir)
    image_cfg = _load_artifact_config(image_dir)

    paired = enumerate_paired(dev_dirs)
    audio_samples = [
        Sample(p.audio_path, p.name, p.label, p.speaker_id, p.session_id)
        for p in paired
    ]
    image_samples = [
        Sample(p.image_path, p.name, p.label, p.speaker_id, p.session_id)
        for p in paired
    ]
    Xa, _, _, _ = extract_set(audio_samples, audio_cfg, "audio")
    Xi, _, _, _ = extract_set(image_samples, image_cfg, "image")
    a_scores = np.asarray(audio_model.estimator.decision_function(Xa), dtype=np.float64)
    i_scores = np.asarray(image_model.estimator.decision_function(Xi), dtype=np.float64)
    fused = score_fusion(fmodel, a_scores, i_scores)
    y = np.asarray([s.label for s in paired])
    return fused, y


def score_dev(
    cfg: Config, artifacts_root: Path, dev_dirs: list[Path] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    dirs = dev_dirs if dev_dirs is not None else list(cfg.data.dev_dirs)
    if cfg.modality in ("audio", "image"):
        return _score_unimodal(cfg, artifacts_root, cfg.modality, dirs)
    if cfg.modality == "fusion":
        return _score_fusion(cfg, artifacts_root, dirs)
    raise ValueError(f"unknown modality: {cfg.modality!r}")


def _eer_point(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float, float]:
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float(fpr[idx]), float(fnr[idx]), float((fpr[idx] + fnr[idx]) / 2)


def _det_warp(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return norm.ppf(np.clip(p, eps, 1 - eps))


def _setup_det_axes(ax: plt.Axes) -> None:
    pos = norm.ppf(np.asarray(DET_TICKS))
    labels = [f"{t * 100:g}" for t in DET_TICKS]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    lo, hi = _det_warp(np.asarray(DET_AXIS_RANGE))
    ax.set_xlim(float(lo), float(hi))
    ax.set_ylim(float(lo), float(hi))
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_ylabel("False Negative Rate (%)")
    ax.set_title("DET")
    ax.grid(True, which="both", alpha=0.3)
    diag = _det_warp(np.asarray(DET_AXIS_RANGE))
    ax.plot(diag, diag, color="0.5", linestyle="--", linewidth=1, alpha=0.6, zorder=0)


def _setup_roc_axes(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.grid(True, alpha=0.3)


def plot_curves(
    systems: list[tuple[str, np.ndarray, np.ndarray]], output_path: Path
) -> None:
    fig, (ax_roc, ax_det) = plt.subplots(1, 2, figsize=(13, 6))
    _setup_roc_axes(ax_roc)
    _setup_det_axes(ax_det)
    cmap = plt.get_cmap("tab10")
    for i, (label, scores, y) in enumerate(systems):
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            print(f"  skipping {label}: only one class present in dev")
            continue
        fpr, tpr, _ = roc_curve(y, scores)
        auc = float(roc_auc_score(y, scores))
        fpr_eer, fnr_eer, eer = _eer_point(fpr, tpr)
        color = cmap(i % 10)

        ax_roc.plot(
            fpr, tpr, color=color, linewidth=1.8, label=f"{label} (AUC={auc:.3f})"
        )
        ax_roc.scatter([fpr_eer], [1 - fnr_eer], color=color, s=45, zorder=5)

        fnr = 1 - tpr
        ax_det.plot(
            _det_warp(fpr),
            _det_warp(fnr),
            color=color,
            linewidth=1.8,
            label=f"{label} (EER={eer * 100:.2f}%)",
        )
        # Clamp EER marker to the visible axis floor so a perfect classifier
        # (eer=0 → probit=-inf) still shows up at the bottom-left corner.
        eer_for_plot = max(eer, DET_AXIS_RANGE[0])
        eer_warp = float(_det_warp(np.asarray([eer_for_plot]))[0])
        ax_det.scatter([eer_warp], [eer_warp], color=color, s=45, zorder=5)

    ax_roc.legend(loc="lower right", fontsize=9)
    ax_det.legend(loc="upper right", fontsize=9)
    fig.suptitle("Dev-split ROC / DET", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/curves.png"))
    ap.add_argument(
        "--dev-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="override config.data.dev_dirs (e.g. point at dataset/*_dev_stress)",
    )
    args = ap.parse_args()

    systems: list[tuple[str, np.ndarray, np.ndarray]] = []
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        dirs = args.dev_dirs if args.dev_dirs else list(cfg.data.dev_dirs)
        print(f"Scoring {cfg.name} ({cfg.modality}) on {[str(d) for d in dirs]} ...")
        scores, y = score_dev(cfg, args.artifacts, dev_dirs=dirs)
        systems.append((cfg.name, scores, y))

    plot_curves(systems, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
