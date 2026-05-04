#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from cv import session_grouped_folds


class _SessionView:
    __slots__ = ("name", "session_id")

    def __init__(self, name: str, session_id: str | None) -> None:
        self.name = name
        self.session_id = session_id


def _parse_session(stem: str) -> str | None:
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]}_{parts[1]}"


def _eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def run_cv(artifact_dir: Path, n_folds: int, c: float) -> None:
    d = np.load(artifact_dir / "train_features.npz", allow_pickle=True)
    X = np.asarray(d["X"])
    y = np.asarray(d["y"]).astype(int)
    names = [str(n) for n in d["names"]]
    samples = [_SessionView(n, _parse_session(n)) for n in names]

    sessions = sorted({s.session_id for s in samples if s.session_id})
    target_sessions = sorted({s.session_id for s, lbl in zip(samples, y) if lbl == 1 and s.session_id})
    nontarget_sessions = sorted({s.session_id for s, lbl in zip(samples, y) if lbl == 0 and s.session_id})

    print(f"Artifact: {artifact_dir.name}")
    print(f"  X shape:           {X.shape}")
    print(f"  target samples:    {(y==1).sum()}  ({len(target_sessions)} sessions: {target_sessions})")
    print(f"  non-target samples:{(y==0).sum()}  ({len(nontarget_sessions)} sessions)")
    print(f"  total sessions:    {len(sessions)}")
    print()

    folds = session_grouped_folds(samples, n_folds)
    fold_aucs = []
    fold_eers = []
    for fold in folds:
        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_dv, y_dv = X[fold.dev_idx], y[fold.dev_idx]

        held_target_sessions = sorted({samples[i].session_id for i in fold.dev_idx if y[i] == 1})
        held_nontarget_sessions = sorted({samples[i].session_id for i in fold.dev_idx if y[i] == 0})

        if (y_tr == 1).sum() == 0 or (y_dv == 1).sum() == 0:
            print(f"  fold {fold.fold_idx}: skipping (no target on one side)")
            continue

        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        clf.fit(X_tr, y_tr)
        s_dv = clf.decision_function(X_dv)
        auc = roc_auc_score(y_dv, s_dv)
        eer = _eer(y_dv, s_dv)
        hard = (s_dv >= 0).astype(int)
        tp = int(((hard == 1) & (y_dv == 1)).sum())
        fp = int(((hard == 1) & (y_dv == 0)).sum())
        fn = int(((hard == 0) & (y_dv == 1)).sum())
        tn = int(((hard == 0) & (y_dv == 0)).sum())
        fold_aucs.append(auc)
        fold_eers.append(eer)
        print(
            f"  fold {fold.fold_idx}: holdout n={len(fold.dev_idx)} "
            f"({(y_dv==1).sum()} tgt / {(y_dv==0).sum()} nontgt) "
            f"-> AUC={auc:.4f}  EER={eer*100:.2f}%  TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}"
        )
        print(f"    target sessions held: {held_target_sessions}")
        print(f"    non-target sessions held ({len(held_nontarget_sessions)}): {held_nontarget_sessions[:6]}{' ...' if len(held_nontarget_sessions)>6 else ''}")
    print()
    print(f"  CV AUC mean:  {np.mean(fold_aucs):.4f}  (per-fold {[f'{a:.4f}' for a in fold_aucs]})")
    print(f"  CV EER mean:  {np.mean(fold_eers)*100:.2f}%  (per-fold {[f'{e*100:.2f}%' for e in fold_eers]})")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-artifact", type=Path, default=Path("artifacts/audio_baseline_logreg"))
    ap.add_argument("--image-artifact", type=Path, default=Path("artifacts/image_baseline_logreg"))
    ap.add_argument("--n-folds", type=int, default=2, help="2 = leave-one-target-session-out")
    args = ap.parse_args()

    print("=" * 70)
    print("AUDIO  (logreg C=1.0, class_weight=balanced - production config)")
    print("=" * 70)
    run_cv(args.audio_artifact, n_folds=args.n_folds, c=1.0)

    print("=" * 70)
    print("IMAGE  (logreg C=0.1, class_weight=balanced - production config)")
    print("=" * 70)
    run_cv(args.image_artifact, n_folds=args.n_folds, c=0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
