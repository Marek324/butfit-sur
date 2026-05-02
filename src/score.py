"""Write the assignment-format result file:

    <segment_name> <score> <hard_decision>

Plain ASCII, one row per segment, three space-separated columns.
hard_decision = 1 (target) if score >= threshold else 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_result_file(
    names: list[str],
    scores: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, s in zip(names, scores, strict=True):
        decision = 1 if float(s) >= threshold else 0
        lines.append(f"{name} {float(s):.6f} {decision}\n")
    path.write_text("".join(lines), encoding="ascii")
