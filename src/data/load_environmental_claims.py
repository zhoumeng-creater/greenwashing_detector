from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import add_config_arg, ensure_dir, load_config, project_path


def _split_name_from_path(path: Path) -> str:
    name = path.stem.split("-")[0]
    return "validation" if name in {"dev", "valid", "val"} else name


def _read_local_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported environmental_claims file type: {path}")


def load_local_environmental_claims_dataset(config: dict | None = None) -> dict[str, pd.DataFrame]:
    config = config or load_config("configs/default.yaml")
    raw_dir = project_path(Path(config["paths"]["raw_dir"]) / "environmental_claims")
    if not raw_dir.exists():
        return {}

    splits: dict[str, pd.DataFrame] = {}
    for pattern in ("*.jsonl", "*.csv", "*.parquet"):
        for path in sorted(raw_dir.glob(pattern)):
            split_name = _split_name_from_path(path)
            splits[split_name] = _read_local_file(path)
    return splits


def load_environmental_claims_dataset(config: dict | None = None, prefer_local: bool = True) -> dict[str, pd.DataFrame]:
    config = config or load_config("configs/default.yaml")
    if prefer_local:
        local_splits = load_local_environmental_claims_dataset(config)
        if local_splits:
            return local_splits

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raw_dir = project_path(Path(config["paths"]["raw_dir"]) / "environmental_claims")
        raise RuntimeError(
            "No local environmental_claims data was found and the optional `datasets` package is not available. "
            f"Place train/dev/test JSONL, CSV, or Parquet files under {raw_dir}."
        ) from exc

    dataset = load_dataset("climatebert/environmental_claims")
    splits: dict[str, pd.DataFrame] = {}
    for split_name, split in dataset.items():
        splits[split_name] = split.to_pandas()
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    args = parser.parse_args()
    config = load_config(args.config)

    out_dir = ensure_dir(Path(config["paths"]["processed_dir"]) / "environmental_claims")
    splits = load_environmental_claims_dataset(config)
    for split_name, frame in splits.items():
        output_path = project_path(out_dir / f"{split_name}.csv")
        frame.to_csv(output_path, index=False)
        print(f"Wrote {split_name}: {len(frame)} rows -> {output_path}")


if __name__ == "__main__":
    main()
