from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.common import add_config_arg, load_config, project_path, resolve_model_ref
from src.data.load_environmental_claims import load_environmental_claims_dataset
from src.models.climatebert_detector import EnvironmentalClaimDetector


def infer_text_label_columns(frame: pd.DataFrame) -> tuple[str, str]:
    text_candidates = ["text", "sentence", "claim", "input"]
    label_candidates = ["label", "labels", "environmental_claim", "is_environmental_claim"]
    text_col = next((col for col in text_candidates if col in frame.columns), frame.columns[0])
    label_col = next((col for col in label_candidates if col in frame.columns), frame.columns[-1])
    return text_col, label_col


def normalize_label(value) -> int:
    if isinstance(value, str):
        lower = value.lower()
        return int(lower in {"1", "true", "yes", "environmental", "claim", "environmental_claim"})
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)

    splits = load_environmental_claims_dataset(config)
    split_name = "test" if "test" in splits else list(splits.keys())[-1]
    frame = splits[split_name]
    if args.limit:
        frame = frame.head(args.limit)
    text_col, label_col = infer_text_label_columns(frame)

    detector = EnvironmentalClaimDetector(
        resolve_model_ref(config, "detector"),
        cache_dir=config["paths"].get("model_cache_dir"),
    )
    y_true = [normalize_label(value) for value in frame[label_col].tolist()]
    y_pred = [int(detector.predict(text).is_environmental_claim) for text in frame[text_col].astype(str).tolist()]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "split": split_name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    output_dir = project_path(Path(config["paths"]["outputs_dir"]) / "runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "claim_detector_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(metrics)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
