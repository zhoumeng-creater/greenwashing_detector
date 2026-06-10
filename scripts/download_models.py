from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import get_configured_model_id, load_config, model_snapshot_dir, project_path


CORE_MODEL_KEYS = ["generator", "embedding", "embedding_fallback", "detector"]
OPTIONAL_MODEL_KEYS = ["prompt_guard"]
MODEL_KEYS = CORE_MODEL_KEYS + OPTIONAL_MODEL_KEYS


def repo_to_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def register_local_model(config: dict, key: str, source: Path, copy: bool) -> dict:
    repo_id = get_configured_model_id(config, key)
    source = source.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {source}")

    target = model_snapshot_dir(config, repo_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        target_resolved = target.resolve()
        if source == target_resolved:
            return {
                "key": key,
                "repo_id": repo_id,
                "source": str(source),
                "target": str(target),
                "mode": "already_registered",
                "file_count": sum(1 for p in target.rglob("*") if p.is_file()),
            }
        if source.is_relative_to(target_resolved):
            raise ValueError(
                f"Refusing to replace {target} because the source path is inside that cache directory: {source}"
            )

    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    if copy:
        shutil.copytree(source, target)
        mode = "copied"
    else:
        target.symlink_to(source, target_is_directory=True)
        mode = "symlinked"

    return {
        "key": key,
        "repo_id": repo_id,
        "source": str(source),
        "target": str(target),
        "mode": mode,
        "file_count": sum(1 for p in target.rglob("*") if p.is_file()),
    }


def download_from_huggingface(config: dict, keys: list[str]) -> list[dict]:
    """Optional helper for the machine that can directly access Hugging Face.

    This intentionally does not use huggingface-cli or hf-mirror. If direct access is
    unavailable, use --*-local options with models downloaded from ModelScope instead.
    """
    from huggingface_hub import snapshot_download

    results = []
    for key in keys:
        repo_id = get_configured_model_id(config, key)
        target = model_snapshot_dir(config, repo_id)
        target.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        results.append(
            {
                "key": key,
                "repo_id": repo_id,
                "target": str(target),
                "downloaded_path": path,
                "file_count": sum(1 for p in target.rglob("*") if p.is_file()),
            }
        )
    return results


def parse_key_list(value: str) -> list[str]:
    if value == "all":
        return CORE_MODEL_KEYS
    if value == "all-with-optional":
        return MODEL_KEYS
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_KEYS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Valid keys: {MODEL_KEYS}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register locally downloaded models, e.g. from ModelScope, into the project offline cache. "
            "This script does not require the 4090D computer to access Hugging Face."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--copy", action="store_true", help="Copy model directories instead of creating symlinks.")
    parser.add_argument("--generator-local", default="", help="Local path for Qwen generator model.")
    parser.add_argument("--embedding-local", default="", help="Local path for embedding model.")
    parser.add_argument("--embedding-fallback-local", default="", help="Local path for fallback embedding model.")
    parser.add_argument("--detector-local", default="", help="Local path for ClimateBERT detector model.")
    parser.add_argument("--prompt-guard-local", default="", help="Local path for Prompt Guard model.")
    parser.add_argument(
        "--prompt-guard-id",
        default="",
        help="Model id used for the Prompt Guard cache directory if configs/default.yaml leaves it blank.",
    )
    parser.add_argument(
        "--download-from-hf",
        default="",
        help="Optional direct Hugging Face download on this computer. Example: generator,embedding,detector, all, or all-with-optional.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompt_guard_id:
        config.setdefault("optional_models", {})["prompt_guard"] = args.prompt_guard_id
    registrations = []
    local_args = {
        "generator": args.generator_local,
        "embedding": args.embedding_local,
        "embedding_fallback": args.embedding_fallback_local,
        "detector": args.detector_local,
        "prompt_guard": args.prompt_guard_local,
    }

    for key, path in local_args.items():
        if path:
            registrations.append(register_local_model(config, key, Path(path), copy=args.copy))

    downloads = []
    if args.download_from_hf:
        keys = parse_key_list(args.download_from_hf)
        downloads = download_from_huggingface(config, keys)

    manifest = {
        "registered_local_models": registrations,
        "downloaded_from_hf": downloads,
        "note": "Transfer offline_bundle/model_cache to the 4090D computer and run without external Hugging Face access.",
    }
    cache_dir = project_path(config["paths"]["model_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
