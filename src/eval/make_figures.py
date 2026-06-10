from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import add_config_arg, load_config, project_path


def save_metric_bar(summary: pd.DataFrame, output: Path, title: str) -> None:
    plt.figure(figsize=(6, 3.5))
    sns.barplot(data=summary, x="metric", y="value")
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--run-dir", default="outputs/runs")
    args = parser.parse_args()
    config = load_config(args.config)

    run_dir = project_path(args.run_dir)
    figures_dir = project_path(Path(config["paths"]["outputs_dir"]) / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    retrieval_path = run_dir / "retrieval_metrics.csv"
    if retrieval_path.exists():
        df = pd.read_csv(retrieval_path)
        summary = df[["recall_at_5", "recall_at_10", "mrr_at_10"]].mean().reset_index()
        summary.columns = ["metric", "value"]
        save_metric_bar(summary, figures_dir / "retrieval_metrics.png", "Retrieval Metrics")

    detector_path = run_dir / "claim_detector_metrics.csv"
    if detector_path.exists():
        df = pd.read_csv(detector_path)
        metric_cols = [col for col in ["accuracy", "precision", "recall", "f1"] if col in df.columns]
        summary = df[metric_cols].iloc[0].reset_index()
        summary.columns = ["metric", "value"]
        save_metric_bar(summary, figures_dir / "claim_detector_metrics.png", "Claim Detector Metrics")

    prompt_guard_path = run_dir / "prompt_guard_metrics.csv"
    if prompt_guard_path.exists():
        df = pd.read_csv(prompt_guard_path)
        metric_cols = [
            col
            for col in ["accuracy", "attack_block_rate", "benign_false_positive_rate"]
            if col in df.columns
        ]
        summary = df[metric_cols].iloc[0].reset_index()
        summary.columns = ["metric", "value"]
        save_metric_bar(summary, figures_dir / "prompt_guard_metrics.png", "Prompt Guard Metrics")


if __name__ == "__main__":
    main()
