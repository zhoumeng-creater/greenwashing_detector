from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import add_config_arg, load_config, project_path, read_jsonl
from src.generation.output_schema import validate_output
from src.generation.rewrite_claim import count_vague_terms


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_jsonl(args.predictions)
    rows = []
    for record in records:
        original = record.get("claim", record.get("input_claim", ""))
        rewrite = record.get("responsible_rewrite", "")
        rows.append(
            {
                "claim_id": record.get("claim_id", ""),
                "schema_pass": int(not validate_output(record)),
                "citation_count": len(record.get("citations", [])),
                "original_vague_terms": count_vague_terms(original),
                "rewrite_vague_terms": count_vague_terms(rewrite),
                "vague_term_reduction": count_vague_terms(original) - count_vague_terms(rewrite),
            }
        )
    output_dir = project_path(Path(config["paths"]["outputs_dir"]) / "runs")
    output_path = output_dir / "rewrite_quality_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(pd.DataFrame(rows).mean(numeric_only=True).to_dict())


if __name__ == "__main__":
    main()
