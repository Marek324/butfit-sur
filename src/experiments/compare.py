from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_runs(metric_files: list[Path]) -> str:
    rows: list[dict[str, Any]] = []
    for p in metric_files:
        if not p.is_file():
            continue
        rows.append(json.loads(p.read_text()))
    if not rows:
        return "(no metrics.json files found)"

    def fmt(v: float | None, w: int = 7, prec: int = 4) -> str:
        return "  n/a  " if v is None else f"{v:>{w}.{prec}f}"

    out = [
        "name                                modality  classifier  n_train  AUC      EER%"
    ]
    out.append("-" * 90)
    for r in rows:
        dev = r.get("dev", {})
        auc = dev.get("auc")
        eer = dev.get("eer")
        eer_pct = None if eer is None else eer * 100.0
        out.append(
            f"{r['name']:<35} {r['modality']:<8} "
            f"{r.get('classifier', '-'):<10} "
            f"{r.get('n_train', '-'):>7}  "
            f"{fmt(auc):>6}  {fmt(eer_pct, prec=2):>6}"
        )
    return "\n".join(out)
