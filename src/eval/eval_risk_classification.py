from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.common import add_config_arg, load_config, project_path, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--predictions", required=True, help="JSONL outputs from analyze_claim.")
    parser.add_argument("--gold", default="data/eval_claims/seed_eval_claims.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    predictions = read_jsonl(args.predictions)
    gold = {row["claim_id"]: row for row in read_jsonl(args.gold)}

    rows = []
    for pred in predictions:
        claim_id = pred.get("claim_id")
        if not claim_id or claim_id not in gold:
            continue
        rows.append(
            {
                "claim_id": claim_id,
                "gold": gold[claim_id]["risk_level"],
                "pred": pred.get("greenwashing_risk_level", "INSUFFICIENT_EVIDENCE"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No matching claim_id values between predictions and gold.")
    metrics = {
        "n": len(frame),
        "accuracy": accuracy_score(frame["gold"], frame["pred"]),
        "macro_f1": f1_score(frame["gold"], frame["pred"], average="macro"),
        "high_risk_recall": (
            ((frame["gold"] == "HIGH") & (frame["pred"] == "HIGH")).sum()
            / max((frame["gold"] == "HIGH").sum(), 1)
        ),
    }
    output_dir = project_path(Path(config["paths"]["outputs_dir"]) / "runs")
    output_path = output_dir / "risk_classification_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(metrics)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
