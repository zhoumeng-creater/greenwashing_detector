from __future__ import annotations

import argparse
from pathlib import Path

from src.common import add_config_arg, load_config, read_jsonl, write_text


def generate_report(records: list[dict]) -> str:
    lines = [
        "# AI Ethics & Compliance Report Draft",
        "",
        "## Scope",
        "",
        "This draft summarizes ethical and compliance risks observed in Greenwashing Detector outputs.",
        "",
        "## Cases",
        "",
    ]
    for i, record in enumerate(records, 1):
        lines.extend(
            [
                f"### Case {i}: {record.get('claim', record.get('input_claim', 'Unknown claim'))}",
                "",
                f"- Risk level: `{record.get('greenwashing_risk_level', 'unknown')}`",
                f"- Risk categories: `{', '.join(record.get('risk_categories', []))}`",
                f"- Citation audit passes: `{record.get('citation_audit', {}).get('passes', 'unknown')}`",
                f"- Ethics audit passes: `{record.get('ethics_audit', {}).get('passes', 'unknown')}`",
                f"- Responsible rewrite: {record.get('responsible_rewrite', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Mitigation Principles",
            "",
            "1. Use risk-indication language, not final legal accusations.",
            "2. Mark evidence gaps clearly.",
            "3. Ground every risk category in cited guidance or evidence.",
            "4. Avoid vague green wording in rewrites.",
            "5. Keep consumer explanations neutral and educational.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--input", required=True, help="JSONL sample outputs.")
    parser.add_argument("--output", default="outputs/demo_samples/ethics_compliance_report.md")
    args = parser.parse_args()
    load_config(args.config)
    records = read_jsonl(args.input)
    report = generate_report(records)
    write_text(Path(args.output), report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
