
import argparse
import json
import statistics as stats
from pathlib import Path

SILERO = Path("scripts/vad_compare/silero_segments.json")
EZ = Path("scripts/vad_compare/energy_zcr_segments.json")
HOP_MS = 10  # discretization for frame-level comparison


def segments_to_mask(
    segments: list[list[float]], dur_s: float, hop_ms: int
) -> list[bool]:
    n = max(1, int(round(dur_s * 1000.0 / hop_ms)))
    mask = [False] * n
    for start, end in segments:
        i0 = max(0, int(round(start * 1000.0 / hop_ms)))
        i1 = min(n, int(round(end * 1000.0 / hop_ms)))
        for i in range(i0, i1):
            mask[i] = True
    return mask


def confusion(ref: list[bool], hyp: list[bool]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for r, h in zip(ref, hyp, strict=True):
        if r and h:
            tp += 1
        elif (not r) and h:
            fp += 1
        elif r and (not h):
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_ref_pos = (tp + fn) / n
    p_hyp_pos = (tp + fp) / n
    pe = p_ref_pos * p_hyp_pos + (1 - p_ref_pos) * (1 - p_hyp_pos)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def split_of(rel_path: str) -> str:
    return rel_path.split("/", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-file", action="store_true", help="print per-file metrics too"
    )
    args = ap.parse_args()

    silero = json.loads(SILERO.read_text())
    ez = json.loads(EZ.read_text())

    common = sorted(set(silero["files"]) & set(ez["files"]))
    only_silero = set(silero["files"]) - set(ez["files"])
    only_ez = set(ez["files"]) - set(silero["files"])
    if only_silero or only_ez:
        print(
            f"WARNING: {len(only_silero)} silero-only, {len(only_ez)} ez-only files (skipped)"
        )

    by_split: dict[str, dict[str, list[float]]] = {}
    grand: dict[str, list[float]] = {
        "silero_ratio": [],
        "ez_ratio": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": [],
        "kappa": [],
        "dur_s": [],
    }

    per_file_rows: list[tuple] = []

    for rel in common:
        s = silero["files"][rel]
        e = ez["files"][rel]
        dur = s["duration_s"]
        ref = segments_to_mask(s["speech_segments_s"], dur, HOP_MS)
        hyp = segments_to_mask(e["speech_segments_s"], dur, HOP_MS)
        # Pad/truncate to equal length (rounding error).
        n = min(len(ref), len(hyp))
        ref = ref[:n]
        hyp = hyp[:n]

        tp, fp, fn, tn = confusion(ref, hyp)
        prec = tp / (tp + fp) if (tp + fp) else (1.0 if tp == 0 and fp == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) else (1.0 if tp == 0 and fn == 0 else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        kappa = cohens_kappa(tp, fp, fn, tn)

        sp = split_of(rel)
        bucket = by_split.setdefault(
            sp,
            {
                "silero_ratio": [],
                "ez_ratio": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "iou": [],
                "kappa": [],
                "dur_s": [],
            },
        )
        for k, v in (
            ("silero_ratio", s["speech_ratio"]),
            ("ez_ratio", e["speech_ratio"]),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1),
            ("iou", iou),
            ("kappa", kappa),
            ("dur_s", dur),
        ):
            bucket[k].append(v)
            grand[k].append(v)

        per_file_rows.append(
            (
                rel,
                dur,
                s["speech_ratio"],
                e["speech_ratio"],
                prec,
                rec,
                iou,
                kappa,
            )
        )

    def fmt_row(name: str, b: dict[str, list[float]]) -> str:
        n = len(b["silero_ratio"])
        return (
            f"{name:<22} n={n:>3}  "
            f"silero={stats.mean(b['silero_ratio']):.3f}  "
            f"ez={stats.mean(b['ez_ratio']):.3f}  "
            f"P={stats.mean(b['precision']):.3f}  "
            f"R={stats.mean(b['recall']):.3f}  "
            f"F1={stats.mean(b['f1']):.3f}  "
            f"IoU={stats.mean(b['iou']):.3f}  "
            f"kappa={stats.mean(b['kappa']):.3f}"
        )

    print()
    print("=" * 110)
    print("Mean per-file metrics (energy+ZCR vs Silero, treating Silero as reference)")
    print("=" * 110)
    for sp in sorted(by_split):
        print(fmt_row(sp, by_split[sp]))
    print("-" * 110)
    print(fmt_row("ALL", grand))
    print()

    # Distribution of disagreement.
    deltas = [
        abs(s - e)
        for s, e in zip(grand["silero_ratio"], grand["ez_ratio"], strict=True)
    ]
    print(
        f"|silero_ratio - ez_ratio| over all files: "
        f"min={min(deltas):.3f} median={stats.median(deltas):.3f} "
        f"mean={stats.mean(deltas):.3f} max={max(deltas):.3f}"
    )

    # Worst kappa files (most disagreement).
    worst = sorted(per_file_rows, key=lambda r: r[7])[:5]
    print("\nWorst 5 files by kappa:")
    print(f"  {'file':<55} dur   silR   ezR    P     R     IoU   kappa")
    for rel, dur, sr, er, p, r, iou, k in worst:
        print(
            f"  {rel:<55} {dur:5.1f} {sr:5.3f} {er:5.3f} {p:5.3f} {r:5.3f} {iou:5.3f} {k:6.3f}"
        )

    if args.per_file:
        print("\nAll files:")
        print(f"  {'file':<55} dur   silR   ezR    P     R     IoU   kappa")
        for rel, dur, sr, er, p, r, iou, k in sorted(per_file_rows):
            print(
                f"  {rel:<55} {dur:5.1f} {sr:5.3f} {er:5.3f} {p:5.3f} {r:5.3f} {iou:5.3f} {k:6.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
