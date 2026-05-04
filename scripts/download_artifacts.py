"""Download artifacts/ (feature cache + per-experiment runs) from HuggingFace.

Run after cloning. Mirrors the HF tree under ``artifacts/`` so ``cache/`` and
each per-experiment run dir land at the same paths your local pipeline expects.
Pass ``--no-cache`` to skip the large ``cache/`` tree (~600 MB).
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Marek324/sur-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cache", action="store_true", help="Skip downloading cache/")
    args = parser.parse_args()

    ignore_patterns = ["*.gitattributes", "README.md", ".gitignore"]
    if args.no_cache:
        ignore_patterns.append("cache/*")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    suffix = " (skipping cache/)" if args.no_cache else ""
    print(f"Downloading artifacts from {REPO_ID}{suffix} …")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(ARTIFACTS),
        ignore_patterns=ignore_patterns,
    )
    ready = "per-experiment runs" if args.no_cache else "cache/ and per-experiment runs"
    print(f"Done. {ready} are ready under artifacts/.")


if __name__ == "__main__":
    sys.exit(main())
