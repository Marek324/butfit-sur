from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from classifier import Model, fit, save, score
from config import Config
from data import (
    PairedSample,
    Sample,
    enumerate_paired,
    enumerate_split,
)
from eval import evaluate, format_metrics

from .extraction import extract_set, feature_config_hash


def _modality_extension(modality: str) -> str:
    return ".wav" if modality == "audio" else ".png"


def _enumerate_unimodal(cfg: Config, modality: str, split: str) -> list[Sample]:
    dirs = cfg.data.train_dirs if split == "train" else cfg.data.dev_dirs
    return enumerate_split(dirs, extension=_modality_extension(modality))


def _feature_names_for(cfg: Config, modality: str, dim: int) -> list[str]:
    if modality == "audio":
        from audio_features import feature_names

        names = feature_names(cfg.audio.features)
        if len(names) != dim:
            return [f"f{i}" for i in range(dim)]
        return names
    return [f"f{i}" for i in range(dim)]


def _metrics_to_jsonable(m: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in asdict(m).items():
        if isinstance(v, tuple):
            d[k] = list(v)
        elif isinstance(v, float) and not np.isfinite(v):
            d[k] = None
        else:
            d[k] = v
    return d


def _load_artifact_config(artifact_dir: Path) -> Config:
    from config import load_config

    return load_config(artifact_dir / "config.toml")


def run_audio_or_image(cfg: Config, out_dir: Path, modality: str) -> dict[str, Any]:
    print(f"=== {cfg.name}  ({modality}) ===")
    train_samples = _enumerate_unimodal(cfg, modality, "train")
    dev_samples = _enumerate_unimodal(cfg, modality, "dev")
    n_t = sum(1 for s in train_samples if s.label == 1)
    print(
        f"train: {len(train_samples)} ({n_t} target, {len(train_samples) - n_t} non-target)"
    )
    print(f"dev:   {len(dev_samples)}")

    print("Extracting train features ...")
    X_tr, y_tr, names_tr, _ = extract_set(train_samples, cfg, modality)
    print(
        f"  X={X_tr.shape}  target={(y_tr == 1).sum()}  non-target={(y_tr == 0).sum()}"
    )

    print("Extracting dev features ...")
    X_dv, y_dv, names_dv, _ = extract_set(dev_samples, cfg, modality)

    print("Fitting classifier ...")
    feat_names = _feature_names_for(cfg, modality, X_tr.shape[1])
    model = fit(X_tr, y_tr, feat_names, cfg.classifier)

    out_dir.mkdir(parents=True, exist_ok=True)
    save(model, out_dir / "model.pkl")
    np.savez(
        out_dir / "train_features.npz",
        X=X_tr,
        y=y_tr,
        names=np.asarray(names_tr),
    )

    s_dv = score(model, X_dv)
    metrics = evaluate(s_dv, y_dv, threshold=cfg.output.hard_threshold_logit)
    print()
    print(format_metrics(metrics))

    payload: dict[str, Any] = {
        "name": cfg.name,
        "modality": modality,
        "classifier": cfg.classifier.type,
        "n_train": int(X_tr.shape[0]),
        "n_train_target": int((y_tr == 1).sum()),
        "n_dev": int(X_dv.shape[0]),
        "n_dev_target": int((y_dv == 1).sum()),
        "feature_dim": int(X_tr.shape[1]),
        "feature_config_hash": feature_config_hash(cfg, modality),
        "dev": _metrics_to_jsonable(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_fusion(cfg: Config, out_dir: Path) -> dict[str, Any]:
    from classifier import load as load_classifier
    from fusion import fit_fusion, save as save_fusion, score_fusion

    print(f"=== {cfg.name}  (fusion) ===")
    if not cfg.fusion.audio_artifact or not cfg.fusion.image_artifact:
        raise ValueError(
            "fusion modality requires fusion.audio_artifact and "
            "fusion.image_artifact to be set in the config"
        )

    artifacts = out_dir.parent
    audio_dir = artifacts / cfg.fusion.audio_artifact
    image_dir = artifacts / cfg.fusion.image_artifact
    audio_model = load_classifier(audio_dir / "model.pkl")
    image_model = load_classifier(image_dir / "model.pkl")
    audio_cfg = _load_artifact_config(audio_dir)
    image_cfg = _load_artifact_config(image_dir)

    if cfg.fusion.train_on != "dev":
        raise ValueError(
            f"fusion.train_on={cfg.fusion.train_on!r} not implemented; only "
            "'dev' is wired up. CV-OOF training is reserved for a follow-up."
        )

    print("Enumerating paired dev samples ...")
    paired = enumerate_paired(cfg.data.dev_dirs)
    print(
        f"  {len(paired)} paired samples ({sum(1 for s in paired if s.label == 1)} target)"
    )

    print("Scoring with audio model ...")
    a_scores, names = _score_paired(paired, audio_cfg, "audio", audio_model)
    print("Scoring with image model ...")
    i_scores, _ = _score_paired(paired, image_cfg, "image", image_model)
    y = np.asarray([s.label for s in paired])

    print("Fitting fusion classifier ...")
    fmodel = fit_fusion(a_scores, i_scores, y, cfg.fusion, cfg.classifier)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fusion(fmodel, out_dir / "model.pkl")
    np.savez(
        out_dir / "train_features.npz",
        audio=a_scores,
        image=i_scores,
        y=y,
        names=np.asarray(names),
    )

    fused = score_fusion(fmodel, a_scores, i_scores)
    metrics = evaluate(fused, y, threshold=cfg.output.hard_threshold_logit)
    print()
    print(format_metrics(metrics))
    print(
        "  (note: dev metrics are biased because fusion was trained on dev; "
        "use train_on='cv_oof' for an unbiased estimate.)"
    )

    payload: dict[str, Any] = {
        "name": cfg.name,
        "modality": "fusion",
        "audio_artifact": cfg.fusion.audio_artifact,
        "image_artifact": cfg.fusion.image_artifact,
        "score_normalize": cfg.fusion.score_normalize,
        "train_on": cfg.fusion.train_on,
        "n_dev": int(len(y)),
        "n_dev_target": int((y == 1).sum()),
        "dev": _metrics_to_jsonable(metrics),
        "weights": np.asarray(fmodel.estimator.coef_).tolist(),
        "intercept": np.asarray(fmodel.estimator.intercept_).tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_experiment(cfg: Config, out_dir: Path) -> dict[str, Any]:
    if cfg.modality in ("audio", "image"):
        return run_audio_or_image(cfg, out_dir, cfg.modality)
    if cfg.modality == "fusion":
        return run_fusion(cfg, out_dir)
    raise ValueError(f"unknown modality: {cfg.modality!r}")


def score_unlabeled(
    cfg: Config, model: Model, samples: list[Sample], modality: str
) -> tuple[np.ndarray, list[str]]:
    X, _y, names, _ = extract_set(samples, cfg, modality)
    return score(model, X), names


def _score_paired(
    paired: list[PairedSample], cfg: Config, modality: str, model: Model
) -> tuple[np.ndarray, list[str]]:
    if modality == "audio":
        unimodal = [
            Sample(
                path=p.audio_path,
                name=p.name,
                label=p.label,
                speaker_id=p.speaker_id,
                session_id=p.session_id,
            )
            for p in paired
        ]
    else:
        unimodal = [
            Sample(
                path=p.image_path,
                name=p.name,
                label=p.label,
                speaker_id=p.speaker_id,
                session_id=p.session_id,
            )
            for p in paired
        ]
    X, _y, names, _ = extract_set(unimodal, cfg, modality)
    return score(model, X), names
