from __future__ import annotations

import argparse
from pathlib import Path

from src.common import add_config_arg, load_config, write_jsonl


DEFAULT_EVAL_CLAIMS = [
    {
        "claim_id": "E001",
        "claim": "Our bottle is 100% eco-friendly and carbon neutral.",
        "product_category": "packaging",
        "jurisdiction": "general",
        "risk_level": "HIGH",
        "risk_categories": ["vague_general_claim", "absolute_claim", "carbon_offset_or_net_zero_risk"],
    },
    {
        "claim_id": "E002",
        "claim": "This package contains 60% recycled paper based on supplier documentation.",
        "product_category": "packaging",
        "jurisdiction": "general",
        "risk_level": "LOW",
        "risk_categories": [],
    },
    {
        "claim_id": "E003",
        "claim": "Our trainers are made with sustainable materials.",
        "product_category": "fashion",
        "jurisdiction": "general",
        "risk_level": "MEDIUM",
        "risk_categories": ["vague_general_claim", "unsubstantiated_claim"],
    },
    {
        "claim_id": "E004",
        "claim": "This product is recyclable everywhere.",
        "product_category": "consumer_goods",
        "jurisdiction": "US",
        "risk_level": "HIGH",
        "risk_categories": ["absolute_claim", "recyclability_ambiguity"],
    },
    {
        "claim_id": "E005",
        "claim": "We will become net zero by 2030, so our products are climate friendly today.",
        "product_category": "general",
        "jurisdiction": "general",
        "risk_level": "HIGH",
        "risk_categories": ["future_target_overclaim", "carbon_offset_or_net_zero_risk"],
    },
    {
        "claim_id": "E006",
        "claim": "Now greener than before.",
        "product_category": "general",
        "jurisdiction": "UK",
        "risk_level": "MEDIUM",
        "risk_categories": ["misleading_comparison", "unsubstantiated_claim"],
    },
    {
        "claim_id": "E007",
        "claim": "Certified green by an independent label.",
        "product_category": "general",
        "jurisdiction": "general",
        "risk_level": "MEDIUM",
        "risk_categories": ["certification_ambiguity"],
    },
    {
        "claim_id": "E008",
        "claim": "Our refill pouch uses 40% less plastic than our previous bottle.",
        "product_category": "packaging",
        "jurisdiction": "general",
        "risk_level": "LOW",
        "risk_categories": [],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = Path(config["paths"]["eval_dir"]) / "seed_eval_claims.jsonl"
    write_jsonl(output_path, DEFAULT_EVAL_CLAIMS)
    print(f"Wrote {len(DEFAULT_EVAL_CLAIMS)} seed evaluation claims to {output_path}")


if __name__ == "__main__":
    main()
