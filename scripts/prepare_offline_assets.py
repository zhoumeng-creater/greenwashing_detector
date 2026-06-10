from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, project_path


def download_guidelines(config: dict) -> list[dict]:
    rules_dir = project_path(config["paths"]["rules_dir"])
    rules_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in config.get("guidance_sources", []):
        source_id = source["source_id"]
        url = source["url"]
        output = rules_dir / f"{source_id}.html"
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "GreenwashingDetector/1.0"})
            response.raise_for_status()
            output.write_text(response.text, encoding="utf-8")
            results.append({"source_id": source_id, "url": url, "status": "downloaded", "path": str(output)})
        except Exception as exc:
            results.append({"source_id": source_id, "url": url, "status": "failed", "error": str(exc)})
    manifest = rules_dir / "guidance_download_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--download-guidelines", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.download_guidelines:
        results = download_guidelines(config)
        ok = sum(1 for row in results if row["status"] == "downloaded")
        print(f"Downloaded {ok}/{len(results)} guidance pages.")
        print("If this fails due to overseas access, the built-in seed rules still allow indexing.")


if __name__ == "__main__":
    main()
