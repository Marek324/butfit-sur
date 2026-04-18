#!/usr/bin/env python3
"""Package the project into a submission ZIP."""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

MODALITY_HINTS = ("audio", "image", "fusion")
EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}


def collect_src(src_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in sorted(src_dir.rglob("*")):
        if path.is_dir():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        rel = path.relative_to(src_dir)
        out.append((path, f"SRC/{rel}"))
    return out


def generate_requirements() -> str:
    """Export runtime deps from pyproject.toml as pip-compatible requirements.txt."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--no-hashes",
            "--no-dev",
            "--no-emit-project",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--docs", type=Path, default=Path("docs/dokumentace.pdf"))
    ap.add_argument("--src", type=Path, default=Path("src"))
    ap.add_argument("--out", type=Path, default=Path("submit.zip"))
    args = ap.parse_args()

    errors: list[str] = []
    result_files: list[Path] = []

    if not args.results.is_dir():
        errors.append(f"results directory not found: {args.results}")
    else:
        result_files = sorted(args.results.glob("*.txt"))
        if not result_files:
            errors.append(f"no *.txt files in {args.results}/")
        elif len(result_files) > 6:
            errors.append(f"too many result files ({len(result_files)}); max 6 allowed")

    if not args.docs.is_file():
        errors.append(f"documentation not found: {args.docs}")

    if not args.src.is_dir():
        errors.append(f"src directory not found: {args.src}")
    elif not any(args.src.rglob("*.py")):
        errors.append(f"no .py files found under {args.src}/")

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1

    hinted = {
        h for h in MODALITY_HINTS if any(h in f.name.lower() for f in result_files)
    }
    missing = set(MODALITY_HINTS) - hinted
    if missing:
        print(
            f"warning: no result filename hints at modality: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )

    src_entries = collect_src(args.src)
    pyproject = Path("pyproject.toml")
    if not pyproject.is_file():
        print(f"error: {pyproject} not found", file=sys.stderr)
        return 1
    requirements = generate_requirements()

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in result_files:
            zf.write(f, f.name)
        zf.write(pyproject, f"SRC/{pyproject.name}")
        zf.writestr("SRC/requirements.txt", requirements)
        for path, arcname in src_entries:
            zf.write(path, arcname)
        zf.write(args.docs, args.docs.name)

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)")
    print(f"  {len(result_files)} result file(s)")
    print(
        f"  SRC/ with {len(src_entries)} source file(s) + pyproject.toml + requirements.txt"
    )
    print(f"  {args.docs.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
