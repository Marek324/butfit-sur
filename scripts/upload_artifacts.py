import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "Marek324/sur-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _experiment_dirs() -> list[Path]:
    if not ARTIFACTS.exists():
        return []
    return [
        child for child in sorted(ARTIFACTS.iterdir())
        if child.is_dir() and child.name not in RESERVED
    ]

# Top-level entries inside artifacts/ that are not per-experiment run dirs.
RESERVED = {"cache", "experiments"}

# Folders where ``--clean`` is supported. ``runs`` cleans every per-experiment dir.
CLEAN_TARGETS = {"cache", "runs"}

RUN_ALLOW = ["config.toml", "metrics.json", "model.pkl", "*.npz"]

def _entries() -> list[tuple[Path, str, list[str] | None, str]]:
    out: list[tuple[Path, str, list[str] | None, str]] = []
    if (ARTIFACTS / "cache").exists():
        out.append((ARTIFACTS / "cache", "cache", None, "cache"))
    for child in sorted(ARTIFACTS.iterdir()) if ARTIFACTS.exists() else []:
        if child.is_dir() and child.name not in RESERVED:
            out.append((child, child.name, ["config.toml", "metrics.json", "model.pkl", "*.npz"], "runs"))
    return out


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="append",
        default=[],
        choices=sorted(CLEAN_TARGETS),
        help="Folder to mirror - remote files not present locally are deleted. "
             "Repeat to clean multiple folders.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip uploading cache/ entirely (useful when only run dirs changed).",
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
        print(f"Clean mode active for: {sorted(clean_set)}")

    cache_dir = ARTIFACTS / "cache"
    if args.no_cache:
        print("  cache/ - skipped (--no-cache)")
    elif not cache_dir.exists() or not any(cache_dir.iterdir()):
        print("  cache/ - empty or missing, skipping")
    else:
        if "cache" in clean_set:
            print("  Deleting remote cache/ (clean mode) …")
            if not args.dry_run:
                try:
                    api.delete_folder(path_in_repo="cache", repo_id=REPO_ID, repo_type=REPO_TYPE)
                except EntryNotFoundError:
                    pass
        print("  Uploading cache/ via upload_large_folder (chunked across commits) …")
        if not args.dry_run:
            # upload_large_folder has no path_in_repo; point it at artifacts/ and
            # filter to cache/** so the local layout maps 1:1 to repo paths.
            api.upload_large_folder(
                repo_id=REPO_ID,
                folder_path=str(ARTIFACTS),
                repo_type=REPO_TYPE,
                allow_patterns=["cache/**"],
            )
        print("  Done: cache/")

    for child in _experiment_dirs():
        if not any(child.iterdir()):
            print(f"  {child.name}/ - empty, skipping")
            continue

        clean_this = "runs" in clean_set
        delete_patterns = ["**"] if clean_this else None

        action = "Mirror-uploading" if clean_this else "Uploading"
        print(f"  {action} {child.name}/ …")
        if args.dry_run:
            print(f"    (dry-run) delete_patterns={delete_patterns}, allow_patterns={RUN_ALLOW}")
            continue
        api.upload_folder(
            folder_path=str(child),
            path_in_repo=child.name,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=RUN_ALLOW,
            delete_patterns=delete_patterns,
        )
        print(f"  Done: {child.name}/")

    print("Upload complete." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    sys.exit(main())
