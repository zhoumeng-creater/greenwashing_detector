from __future__ import annotations

import argparse

from src.common import add_config_arg, load_config, project_path, read_jsonl, write_jsonl
from src.generation.analyze_claim import analyze_claim


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--gold", default="data/eval_claims/seed_eval_claims.jsonl")
    parser.add_argument("--output", default="outputs/demo_samples/eval_predictions.jsonl")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    gold_records = read_jsonl(args.gold)

    results = []
    for record in gold_records:
        result = analyze_claim(
            config=config,
            claim=record["claim"],
            product_category=record.get("product_category", ""),
            jurisdiction=record.get("jurisdiction", ""),
            use_llm=not args.no_llm,
        )
        result["claim_id"] = record["claim_id"]
        results.append(result)
        print(f"[{record['claim_id']}] {result.get('greenwashing_risk_level')}")

    output_path = project_path(args.output)
    write_jsonl(output_path, results)
    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
