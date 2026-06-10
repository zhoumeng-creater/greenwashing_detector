from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import add_config_arg, load_config, project_path, read_jsonl
from src.data.build_eval_claims import DEFAULT_EVAL_CLAIMS
from src.retrieval.search import FaissRetriever, build_query


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--eval-file", default="")
    args = parser.parse_args()
    config = load_config(args.config)

    claims = read_jsonl(args.eval_file) if args.eval_file else DEFAULT_EVAL_CLAIMS
    retriever = FaissRetriever(config)
    rows = []
    for claim in claims:
        query = build_query(
            claim["claim"],
            product_category=claim.get("product_category", ""),
            jurisdiction=claim.get("jurisdiction", ""),
        )
        hits = retriever.search(query, top_k=10)
        expected = set(claim.get("risk_categories", []))
        hit_types = [hit.chunk.get("claim_type") for hit in hits]
        ranks = [i + 1 for i, claim_type in enumerate(hit_types) if claim_type in expected]
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "expected_categories": ",".join(sorted(expected)),
                "top_types": ",".join(hit_types[:5]),
                "recall_at_5": int(any(t in expected for t in hit_types[:5])) if expected else 1,
                "recall_at_10": int(any(t in expected for t in hit_types[:10])) if expected else 1,
                "mrr_at_10": (1 / min(ranks)) if ranks else 0.0,
            }
        )
    output_dir = project_path(Path(config["paths"]["outputs_dir"]) / "runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(pd.DataFrame(rows).mean(numeric_only=True).to_dict())


if __name__ == "__main__":
    main()
