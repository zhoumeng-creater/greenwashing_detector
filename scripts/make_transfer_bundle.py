from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}


def should_include(path: Path, include_models: bool) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return False
    if not include_models and "model_cache" in parts:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-model-cache", action="store_true")
    parser.add_argument("--output", default="greenwashing_detector_transfer.tar.gz")
    args = parser.parse_args()

    output = ROOT.parent / args.output
    if output.exists():
        output.unlink()

    with tarfile.open(output, "w:gz") as tar:
        for path in ROOT.rglob("*"):
            if path == output:
                continue
            if should_include(path, args.include_model_cache):
                tar.add(path, arcname=Path("greenwashing_detector") / path.relative_to(ROOT))

    print(f"Wrote transfer bundle: {output}")
    if not args.include_model_cache:
        print("Model cache excluded. Add --include-model-cache if you already downloaded models and want to transfer them.")


if __name__ == "__main__":
    main()
