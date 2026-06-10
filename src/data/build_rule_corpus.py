from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup

from src.common import add_config_arg, ensure_dir, load_config, project_path, write_jsonl
from src.data.preprocess import chunk_text, clean_text


SEED_RULES = [
    {
        "source_id": "FTC_SUMMARY",
        "jurisdiction": "US",
        "claim_type": "vague_general_claim",
        "title": "General environmental benefit claims",
        "url": "https://www.ftc.gov/business-guidance/resources/environmental-claims-summary-green-guides",
        "text": (
            "Broad, unqualified general environmental benefit claims can be difficult to substantiate. "
            "Marketers should qualify green, eco-friendly, and similar claims with specific environmental "
            "benefits and avoid implying broader benefits than they can support."
        ),
    },
    {
        "source_id": "FTC_SUMMARY",
        "jurisdiction": "US",
        "claim_type": "recyclability_ambiguity",
        "title": "Recyclable claims",
        "url": "https://www.ftc.gov/business-guidance/resources/environmental-claims-summary-green-guides",
        "text": (
            "A recyclable claim should be qualified when recycling facilities are not available to a "
            "substantial majority of consumers or communities where the item is sold."
        ),
    },
    {
        "source_id": "FTC_SUMMARY",
        "jurisdiction": "US",
        "claim_type": "certification_ambiguity",
        "title": "Certifications and seals of approval",
        "url": "https://www.ftc.gov/business-guidance/resources/environmental-claims-summary-green-guides",
        "text": (
            "Certifications or seals may imply environmental benefits. Marketers should disclose material "
            "connections and explain the basis for certification when it is not clear."
        ),
    },
    {
        "source_id": "CMA_GREEN_CLAIMS",
        "jurisdiction": "UK",
        "claim_type": "unsubstantiated_claim",
        "title": "Green claims should be truthful and accurate",
        "url": "https://www.gov.uk/government/publications/green-claims-code-making-environmental-claims",
        "text": (
            "Green claims should be truthful, accurate, clear, unambiguous, and supported by evidence. "
            "They should not omit or hide important information and should consider the full life cycle "
            "where relevant."
        ),
    },
    {
        "source_id": "CMA_GREEN_CLAIMS",
        "jurisdiction": "UK",
        "claim_type": "hidden_tradeoff",
        "title": "Full life cycle and omitted information",
        "url": "https://www.gov.uk/government/publications/green-claims-code-making-environmental-claims",
        "text": (
            "Claims should not focus on a narrow environmental benefit if the overall impact or life-cycle "
            "context would make the claim misleading."
        ),
    },
    {
        "source_id": "EU_GREEN_CLAIMS",
        "jurisdiction": "EU",
        "claim_type": "unsubstantiated_claim",
        "title": "Substantiating environmental claims",
        "url": "https://environment.ec.europa.eu/topics/circular-economy-topics/green-claims_en",
        "text": (
            "Environmental claims should be substantiated, reliable, comparable, and verifiable. Vague or "
            "poorly supported claims risk misleading consumers."
        ),
    },
    {
        "source_id": "ASA_ENV_CLAIMS",
        "jurisdiction": "UK",
        "claim_type": "misleading_comparison",
        "title": "Environmental comparisons",
        "url": "https://www.asa.org.uk/advice-online/environmental-claims-general.html",
        "text": (
            "Environmental comparisons should make the basis of comparison clear. Claims such as greener "
            "or more sustainable can mislead when the compared baseline, scope, or evidence is unclear."
        ),
    },
    {
        "source_id": "LOCAL_RUBRIC",
        "jurisdiction": "general",
        "claim_type": "absolute_claim",
        "title": "Absolute green claims",
        "url": "local://greenwashing-risk-rubric",
        "text": (
            "Absolute expressions such as 100% eco-friendly, zero impact, or completely sustainable are "
            "high risk unless all relevant scope, boundaries, and evidence are provided."
        ),
    },
    {
        "source_id": "LOCAL_RUBRIC",
        "jurisdiction": "general",
        "claim_type": "carbon_offset_or_net_zero_risk",
        "title": "Carbon neutral and offset claims",
        "url": "local://greenwashing-risk-rubric",
        "text": (
            "Carbon neutral, net zero, and offset-based claims should explain the scope, calculation method, "
            "emissions boundary, reduction actions, and role of offsets."
        ),
    },
    {
        "source_id": "LOCAL_RUBRIC",
        "jurisdiction": "general",
        "claim_type": "future_target_overclaim",
        "title": "Future target overclaims",
        "url": "local://greenwashing-risk-rubric",
        "text": (
            "A future environmental target should not be presented as an achieved present benefit. Claims "
            "should distinguish commitments from completed outcomes."
        ),
    },
]


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def load_downloaded_guidance(config: dict) -> list[dict]:
    rules_dir = project_path(config["paths"]["rules_dir"])
    records: list[dict] = []
    for source in config.get("guidance_sources", []):
        source_id = source["source_id"]
        candidates = [
            rules_dir / f"{source_id}.txt",
            rules_dir / f"{source_id}.html",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            text = html_to_text(candidate) if candidate.suffix == ".html" else clean_text(candidate.read_text(encoding="utf-8"))
            if text:
                records.append(
                    {
                        "source_id": source_id,
                        "jurisdiction": source.get("jurisdiction", "unknown"),
                        "claim_type": "official_guidance",
                        "title": source.get("title", source_id),
                        "url": source.get("url", ""),
                        "text": text,
                    }
                )
            break
    return records


def build_rule_chunks(config: dict, include_downloaded: bool = True) -> list[dict]:
    retrieval_config = config["retrieval"]
    records = list(SEED_RULES)
    if include_downloaded:
        records.extend(load_downloaded_guidance(config))

    chunks: list[dict] = []
    for i, record in enumerate(records):
        doc_id = f"R{i:04d}"
        for chunk in chunk_text(
            doc_id=doc_id,
            source_id=record["source_id"],
            title=record["title"],
            jurisdiction=record.get("jurisdiction", "unknown"),
            claim_type=record.get("claim_type", "unknown"),
            text=record["text"],
            url=record.get("url", ""),
            chunk_size_chars=retrieval_config["chunk_size_chars"],
            chunk_overlap_chars=retrieval_config["chunk_overlap_chars"],
        ):
            row = asdict(chunk)
            row["retrieval_text"] = (
                f"{row['title']}\nJurisdiction: {row['jurisdiction']}\n"
                f"Claim type: {row['claim_type']}\n{row['text']}"
            )
            chunks.append(row)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--no-downloaded", action="store_true", help="Use only built-in seed rules.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dir(config["paths"]["processed_dir"])
    chunks = build_rule_chunks(config, include_downloaded=not args.no_downloaded)
    output_path = Path(config["paths"]["processed_dir"]) / "rule_chunks.jsonl"
    write_jsonl(output_path, chunks)
    print(f"Wrote {len(chunks)} rule chunks to {output_path}")


if __name__ == "__main__":
    main()
