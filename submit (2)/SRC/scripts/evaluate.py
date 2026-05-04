
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  

from classifier import load as load_classifier  
from config import Config, load_config  
from data import enumerate_paired, enumerate_split  
from eval import evaluate, format_metrics  
from experiments import extract_set, score_unlabeled  
from fusion import load as load_fusion, score_fusion  
from score import write_result_file  


def _modality_extension(modality: str) -> str:
    return ".wav" if modality == "audio" else ".png"


def _load_artifact_config(d: Path) -> Config:
    return load_config(d / "config.toml")


def _resolve_unlabeled_dirs(cfg: Config, args: argparse.Namespace) -> list[Path]:
    if args.input is not None:
        return [args.input]
    if not cfg.data.eval_dirs:
        raise SystemExit(
            "no eval dirs available: set data.eval_dirs in the config or pass --input DIR"
        )
    return list(cfg.data.eval_dirs)


def evaluate_unimodal(cfg: Config, args: argparse.Namespace, modality: str) -> None:
    model = load_classifier(args.artifacts / cfg.name / "model.pkl")
    ext = _modality_extension(modality)

    if args.split == "dev":
        samples = enumerate_split(cfg.data.dev_dirs, extension=ext)
        scores, _names = score_unlabeled(cfg, model, samples, modality)
        y = np.asarray([s.label for s in samples])
        m = evaluate(scores, y, threshold=cfg.output.hard_threshold_logit)
        print()
        print(format_metrics(m))
        _print_score_distribution(scores, y)
        return

    dirs = _resolve_unlabeled_dirs(cfg, args)
    samples = enumerate_split(dirs, extension=ext, label_required=False)
    scores, names = score_unlabeled(cfg, model, samples, modality)
    out = args.results_dir / cfg.output.result_filename
    write_result_file(names, scores, cfg.output.hard_threshold_logit, out)
    n_pos = int((scores >= cfg.output.hard_threshold_logit).sum())
    print(f"Wrote {out}: {len(names)} segments  ({n_pos} predicted target)")


def evaluate_fusion(cfg: Config, args: argparse.Namespace) -> None:
    fmodel = load_fusion(args.artifacts / cfg.name / "model.pkl")
    audio_dir = args.artifacts / cfg.fusion.audio_artifact
    image_dir = args.artifacts / cfg.fusion.image_artifact
    audio_model = load_classifier(audio_dir / "model.pkl")
    image_model = load_classifier(image_dir / "model.pkl")
    audio_cfg = _load_artifact_config(audio_dir)
    image_cfg = _load_artifact_config(image_dir)

    if args.split == "dev":
        paired = enumerate_paired(cfg.data.dev_dirs)
    else:
        dirs = _resolve_unlabeled_dirs(cfg, args)
        paired = enumerate_paired(dirs, label_required=False)

    print(f"  paired: {len(paired)} samples")
    audio_samples, image_samples = _split_paired(paired)

    Xa, _, names, _ = extract_set(audio_samples, audio_cfg, "audio")
    Xi, _, _, _ = extract_set(image_samples, image_cfg, "image")
    a_scores = np.asarray(audio_model.estimator.decision_function(Xa), dtype=np.float64)
    i_scores = np.asarray(image_model.estimator.decision_function(Xi), dtype=np.float64)
    fused = score_fusion(fmodel, a_scores, i_scores)

    if args.split == "dev":
        y = np.asarray([s.label for s in paired])
        m = evaluate(fused, y, threshold=cfg.output.hard_threshold_logit)
        print()
        print(format_metrics(m))
        _print_score_distribution(fused, y)
        return

    out = args.results_dir / cfg.output.result_filename
    write_result_file(names, fused, cfg.output.hard_threshold_logit, out)
    n_pos = int((fused >= cfg.output.hard_threshold_logit).sum())
    print(f"Wrote {out}: {len(names)} segments  ({n_pos} predicted target)")


def _split_paired(paired):
    from data import Sample

    audio = [
        Sample(p.audio_path, p.name, p.label, p.speaker_id, p.session_id)
        for p in paired
    ]
    image = [
        Sample(p.image_path, p.name, p.label, p.speaker_id, p.session_id)
        for p in paired
    ]
    return audio, image


def _print_score_distribution(scores: np.ndarray, y: np.ndarray) -> None:
    print("\nScore distribution:")
    for lab, lab_name in [(1, "target  "), (0, "non-tgt ")]:
        s_lab = scores[y == lab]
        if len(s_lab) > 0:
            print(
                f"  {lab_name} n={len(s_lab):3d}  "
                f"min={s_lab.min():+7.3f}  med={float(np.median(s_lab)):+7.3f}  "
                f"max={s_lab.max():+7.3f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--split",
        choices=["dev", "eval"],
        default=None,
        help="dev = compute metrics; eval = score config.data.eval_dirs and write the result file",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="score an arbitrary unlabeled directory (overrides eval_dirs)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="where to write the result file in eval / --input mode",
    )
    args = ap.parse_args()

    if args.split is None and args.input is None:
        ap.error("specify --split {dev,eval} or --input DIR")
    if args.split == "dev" and args.input is not None:
        ap.error("--input is not used with --split dev")

    cfg = load_config(args.config)
    if cfg.modality in ("audio", "image"):
        evaluate_unimodal(cfg, args, cfg.modality)
    elif cfg.modality == "fusion":
        evaluate_fusion(cfg, args)
    else:
        raise ValueError(f"unknown modality: {cfg.modality!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
