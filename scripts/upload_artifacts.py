"""Upload artifacts/ (feature cache + per-experiment runs) to HuggingFace.

Default mode is additive — files on HF that no longer exist locally are kept.
Pass ``--clean cache`` (and/or ``runs``) to atomically delete remote files in
those folders that aren't in the local copy.

Folder map:
  artifacts/cache/        → expensive feature cache (audio, audio_pre, image)
  artifacts/<exp_name>/   → per-experiment run dirs (config, metrics, model, features)

Plots (artifacts/*.png, artifacts/experiments/*.png) are skipped — regenerate
locally from the run dirs.

Examples:
  python scripts/upload_artifacts.py                              # additive upload
  python scripts/upload_artifacts.py --clean cache                # mirror cache/
  python scripts/upload_artifacts.py --clean cache --clean runs
  python scripts/upload_artifacts.py --clean cache --dry-run      # preview only
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/sur-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Top-level entries inside artifacts/ that are not per-experiment run dirs.
RESERVED = {"cache", "experiments"}

# Folders where ``--clean`` is supported. ``runs`` cleans every per-experiment dir.
CLEAN_TARGETS = {"cache", "runs"}


def _entries() -> list[tuple[Path, str, list[str] | None, str]]:
    """(local_path, path_in_repo, allow_patterns, clean_key)."""
    out: list[tuple[Path, str, list[str] | None, str]] = []
    if (ARTIFACTS / "cache").exists():
        out.append((ARTIFACTS / "cache", "cache", None, "cache"))
    for child in sorted(ARTIFACTS.iterdir()) if ARTIFACTS.exists() else []:
        if child.is_dir() and child.name not in RESERVED:
            out.append((child, child.name, ["config.toml", "metrics.json", "model.pkl", "*.npz"], "runs"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--clean",
        action="append",
        default=[],
        choices=sorted(CLEAN_TARGETS),
        help="Folder to mirror — remote files not present locally are deleted "
             "atomically with the upload. Repeat to clean multiple folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or deleting.",
    )
    args = parser.parse_args()

    clean_set = set(args.clean)

    api = HfApi()
    if not args.dry_run:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")
    if clean_set:
        print(f"Clean mode active for: {sorted(clean_set)} (remote-only files in those folders will be deleted)")

    for local, path_in_repo, allow_patterns, clean_key in _entries():
        if not local.exists() or not any(local.iterdir()):
            print(f"  {path_in_repo}/  — empty or missing, skipping")
            continue

        clean_this = clean_key in clean_set
        delete_patterns = ["**"] if clean_this else None

        action = "Mirror-uploading" if clean_this else "Uploading"
        print(f"  {action} {path_in_repo}/ …")
        if args.dry_run:
            print(f"    (dry-run) delete_patterns={delete_patterns}, allow_patterns={allow_patterns}")
            continue
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=allow_patterns,
            delete_patterns=delete_patterns,
        )
        print(f"  Done: {path_in_repo}/")

    print("Upload complete." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    sys.exit(main())
