

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


PDF_CSS = """
@page { size: A4; margin: 1.8cm; }
body { font-family: serif; font-size: 10pt; line-height: 1.35; }
header.title { text-align: center; margin-bottom: 1.2em; }
header.title .t { font-size: 16pt; font-weight: bold; }
header.title .s { font-size: 11pt; font-style: italic; margin-top: 0.2em; }
header.title .a { font-size: 10pt; margin-top: 0.4em; }
header.title .d { font-size: 9pt; margin-top: 0.2em; color: #444; }
h1 { font-size: 13pt; margin-top: 0.8em; }
h2 { font-size: 12pt; margin-top: 0.6em; }
h3 { font-size: 11pt; margin-top: 0.4em; }
table { border-collapse: collapse; margin: 0.5em 0; font-size: 9pt; }
th, td { border: 1px solid #777; padding: 2px 6px; }
th { background: #eee; }
pre, code { font-family: monospace; font-size: 8.5pt; }
pre { background: #f4f4f4; padding: 4px 6px; }
"""


def _parse_yaml_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm = text[3:end]
    body = text[end + 3 :].lstrip()
    meta: dict[str, list[str] | str] = {}
    current_key: str | None = None
    for raw in fm.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith(("  - ", "- ")):
            value = line.split("- ", 1)[1].strip().strip('"').strip("'")
            if current_key is None:
                continue
            existing = meta.get(current_key)
            if isinstance(existing, list):
                existing.append(value)
            else:
                meta[current_key] = [value] if existing is None else [existing, value]
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                meta[key] = value.strip('"').strip("'")
            else:
                meta[key] = []
    return meta, body


def _title_block_html(meta: dict) -> str:
    title = meta.get("title", "")
    subtitle = meta.get("subtitle", "")
    authors = meta.get("author", [])
    if isinstance(authors, str):
        authors = [authors]
    date = meta.get("date", "")
    if not (title or subtitle or authors or date):
        return ""
    parts = ['<header class="title">']
    if title:
        parts.append(f'<div class="t">{title}</div>')
    if subtitle:
        parts.append(f'<div class="s">{subtitle}</div>')
    if authors:
        parts.append(f'<div class="a">{" &middot; ".join(authors)}</div>')
    if date:
        parts.append(f'<div class="d">{date}</div>')
    parts.append("</header>")
    return "".join(parts)


def render_md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    import markdown
    from weasyprint import HTML

    text = md_path.read_text()
    meta, body = _parse_yaml_front_matter(text)
    title_html = _title_block_html(meta)
    html_body = markdown.markdown(body, extensions=["tables", "fenced_code"])
    full = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{PDF_CSS}</style></head><body>"
        f"{title_html}{html_body}</body></html>"
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=full, base_url=str(md_path.parent)).write_pdf(str(pdf_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--docs", type=Path, default=Path("docs/dokumentace.pdf"))
    ap.add_argument(
        "--docs-md",
        type=Path,
        default=Path("docs/dokumentace.md"),
        help="Markdown source; rendered to --docs if newer (or if --docs missing)",
    )
    ap.add_argument(
        "--skip-render",
        action="store_true",
        help="use existing --docs as-is, do not render from --docs-md",
    )
    ap.add_argument("--src", type=Path, default=Path("src"))
    ap.add_argument("--scripts-dir", type=Path, default=Path("scripts"))
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("submit.zip"))
    args = ap.parse_args()

    if not args.skip_render and args.docs_md.is_file():
        needs_render = (
            not args.docs.is_file()
            or args.docs_md.stat().st_mtime > args.docs.stat().st_mtime
        )
        if needs_render:
            print(f"Rendering {args.docs_md} -> {args.docs} ...")
            try:
                render_md_to_pdf(args.docs_md, args.docs)
            except ImportError as e:
                print(
                    f"error: PDF render requires markdown + weasyprint ({e}); "
                    f"install them or pass --skip-render with a pre-built {args.docs}",
                    file=sys.stderr,
                )
                return 1

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
