#!/usr/bin/env python3
"""Bootstrap the dataset under ./dataset/."""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2025-2026/SUR_projekt2025-2026.zip"
DEST = Path("dataset")
REQUIRED = ("target_train", "target_dev", "non_target_train", "non_target_dev")


def already_populated() -> bool:
    return all((DEST / d).is_dir() and any((DEST / d).iterdir()) for d in REQUIRED)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if already_populated() and not args.force:
        print(f"{DEST}/ already populated, skipping. Use --force to re-download.")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        zip_path = tdp / "dataset.zip"

        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  got {size_mb:.1f} MB")

        extract_dir = tdp / "extract"
        extract_dir.mkdir()
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        DEST.mkdir(exist_ok=True)
        for d in REQUIRED:
            src = extract_dir / d
            if not src.is_dir():
                raise RuntimeError(f"missing {d} in archive")
            target = DEST / d
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(src), str(target))
        print(f"Placed {len(REQUIRED)} directories under {DEST}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
