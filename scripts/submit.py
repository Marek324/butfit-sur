#!/usr/bin/env python3

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

MODALITY_HINTS = ("audio", "image", "fusion")
EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}

PRODUCTION_SCRIPTS = (
    "augment_dataset.py",
    "train.py",
    "evaluate.py",
    "audio_gmm_pipeline.py",
    "audio_gmm_hybrid.py",
    "fix_vad_empty.py",
    "fusion_hybrid_image.py",
    "fusion_3way.py",
)
PRODUCTION_CONFIGS = (
    "audio_baseline.toml",
    "image_baseline.toml",
    "fusion_baseline.toml",
    "augment_audio.toml",
    "augment_image.toml",
)


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


def collect_scripts(scripts_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for name in PRODUCTION_SCRIPTS:
        p = scripts_dir / name
        if not p.is_file():
            raise SystemExit(f"required script missing: {p}")
        out.append((p, f"SRC/scripts/{name}"))
    return out


def collect_configs(configs_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for name in PRODUCTION_CONFIGS:
        p = configs_dir / name
        if not p.is_file():
            raise SystemExit(f"required config missing: {p}")
        out.append((p, f"SRC/configs/{name}"))
    return out


def generate_requirements() -> str:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--docs", type=Path, default=Path("docs/dokumentace.pdf"))
    ap.add_argument("--src", type=Path, default=Path("src"))
    ap.add_argument("--scripts-dir", type=Path, default=Path("scripts"))
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
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
    script_entries = collect_scripts(args.scripts_dir)
    config_entries = collect_configs(args.configs_dir)
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
        for path, arcname in script_entries:
            zf.write(path, arcname)
        for path, arcname in config_entries:
            zf.write(path, arcname)
        zf.write(args.docs, args.docs.name)

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)")
    print(f"  {len(result_files)} result file(s)")
    print(
        f"  SRC/ with {len(src_entries)} source file(s) "
        f"+ {len(script_entries)} script(s) + {len(config_entries)} config(s) "
        f"+ pyproject.toml + requirements.txt"
    )
    print(f"  {args.docs.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
