from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = project_path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent = config.get("extends")
    if parent:
        base = load_config(parent)
        config = deep_merge(base, {k: v for k, v in config.items() if k != "extends"})
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = project_path(path)
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_text(path: str | Path) -> str:
    path = project_path(path)
    return path.read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def timestamped_run_dir(config: dict[str, Any], name: str) -> Path:
    from datetime import datetime

    root = ensure_dir(Path(config["paths"]["outputs_dir"]) / "runs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")


def get_hf_cache_env(config: dict[str, Any]) -> dict[str, str]:
    cache_dir = project_path(config["paths"].get("model_cache_dir", "offline_bundle/model_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    endpoint = config.get("network", {}).get("hf_endpoint")
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    env.setdefault("HF_HOME", str(cache_dir))
    env.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    return env


def configure_network(config: dict[str, Any]) -> None:
    network = config.get("network", {})
    endpoint = network.get("hf_endpoint")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if network.get("disable_ssl_verify"):
        try:
            from huggingface_hub.utils._http import set_client_factory
            import httpx

            set_client_factory(lambda: httpx.Client(
                follow_redirects=True,
                timeout=None,
                verify=False,
            ))
        except Exception:
            pass


def model_snapshot_dir(config: dict[str, Any], repo_id: str) -> Path:
    cache_dir = project_path(config["paths"].get("model_cache_dir", "offline_bundle/model_cache"))
    return cache_dir / "snapshots" / repo_id.replace("/", "__")


def get_configured_model_id(config: dict[str, Any], model_key: str, required: bool = True) -> str:
    model_id = config.get("models", {}).get(model_key) or config.get("optional_models", {}).get(model_key) or ""
    if required and not model_id:
        raise KeyError(f"No model configured for key: {model_key}")
    return str(model_id)


def resolve_model_ref(config: dict[str, Any], model_key: str, required: bool = True) -> str:
    """Return local snapshot path when present, otherwise the configured repo id.

    This is critical for the offline 4090D machine: from_pretrained() should receive
    a local directory instead of trying to contact Hugging Face.
    """
    repo_id = get_configured_model_id(config, model_key, required=required)
    if not repo_id:
        return ""

    candidate = Path(repo_id).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    project_candidate = project_path(candidate)
    if project_candidate.exists():
        return str(project_candidate)

    local_dir = model_snapshot_dir(config, repo_id)
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    return repo_id


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
