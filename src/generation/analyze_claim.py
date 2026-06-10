from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common import add_config_arg, load_config, project_path, resolve_model_ref, write_jsonl
from src.generation.output_schema import extract_json, fallback_output, validate_output
from src.generation.prompts import build_analysis_prompt, format_context
from src.models.climatebert_detector import EnvironmentalClaimDetector
from src.models.load_qwen import GenerationConfig, QwenGenerator
from src.retrieval.search import FaissRetriever, build_query
from src.safety.citation_audit import citation_audit
from src.safety.ethics_rubric import apply_ethics_rubric
from src.safety.prompt_guard import build_prompt_guard


def analyze_claim(
    *,
    config: dict[str, Any],
    claim: str,
    product_category: str = "",
    jurisdiction: str = "",
    evidence_text: str = "",
    use_llm: bool = True,
    prompt_guard_mode: str | None = None,
    prompt_guard_model: str = "",
) -> dict[str, Any]:
    guard = build_prompt_guard(config, mode=prompt_guard_mode, model_ref=prompt_guard_model)
    guard_result = guard("\n".join(part for part in [claim, evidence_text] if part))
    if guard_result["is_suspicious"]:
        record = fallback_output(claim, "Prompt Guard blocked the input as suspicious.")
        record["prompt_guard"] = guard_result
        record["detector"] = None
        record["retrieved_chunks"] = []
        record["citation_audit"] = citation_audit(record, record["retrieved_chunks"])
        record["ethics_audit"] = apply_ethics_rubric(record)
        return record

    detector = EnvironmentalClaimDetector(
        resolve_model_ref(config, "detector"),
        cache_dir=config["paths"].get("model_cache_dir"),
    )
    detector_result = detector.predict(claim)

    retriever = FaissRetriever(config)
    query = build_query(claim, product_category=product_category, jurisdiction=jurisdiction)
    hits = retriever.search(query)
    context = format_context(hits)

    if not detector_result.is_environmental_claim:
        record = fallback_output(claim, "Detector classified this text as not an environmental claim.")
        record["is_environmental_claim"] = False
    elif use_llm:
        generator = QwenGenerator(
            resolve_model_ref(config, "generator"),
            cache_dir=config["paths"].get("model_cache_dir"),
            load_in_4bit=config["generation"].get("load_in_4bit", True),
        )
        prompt = build_analysis_prompt(
            claim=claim,
            product_category=product_category,
            jurisdiction=jurisdiction,
            evidence_text=evidence_text,
            context=context,
        )
        text = generator.generate(
            prompt,
            GenerationConfig(
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
            ),
        )
        try:
            record = extract_json(text)
            errors = validate_output(record)
            if errors:
                record = fallback_output(claim, "; ".join(errors))
                record["raw_generation"] = text
        except Exception as exc:
            record = fallback_output(claim, str(exc))
            record["raw_generation"] = text
    else:
        record = fallback_output(claim, "LLM disabled; retrieval-only smoke result.")
        record["greenwashing_risk_level"] = "INSUFFICIENT_EVIDENCE"

    record["detector"] = {
        "is_environmental_claim": detector_result.is_environmental_claim,
        "score": detector_result.score,
        "raw_label": detector_result.raw_label,
    }
    record["prompt_guard"] = guard_result
    record["retrieved_chunks"] = [
        {
            "score": hit.score,
            "chunk_id": hit.chunk["chunk_id"],
            "source_id": hit.chunk["source_id"],
            "title": hit.chunk["title"],
            "claim_type": hit.chunk["claim_type"],
            "url": hit.chunk["url"],
        }
        for hit in hits
    ]
    record["citation_audit"] = citation_audit(record, record["retrieved_chunks"])
    record["ethics_audit"] = apply_ethics_rubric(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--product-category", default="")
    parser.add_argument("--jurisdiction", default="")
    parser.add_argument("--evidence-text", default="")
    parser.add_argument("--no-llm", action="store_true", help="Run detector and retrieval only.")
    parser.add_argument("--prompt-guard-mode", choices=["auto", "model", "rule"], default="")
    parser.add_argument("--prompt-guard-model", default="", help="Optional local Prompt Guard directory override.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    record = analyze_claim(
        config=config,
        claim=args.claim,
        product_category=args.product_category,
        jurisdiction=args.jurisdiction,
        evidence_text=args.evidence_text,
        use_llm=not args.no_llm,
        prompt_guard_mode=args.prompt_guard_mode or None,
        prompt_guard_model=args.prompt_guard_model,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.output:
        write_jsonl(Path(args.output), [record])
        print(f"Wrote output to {project_path(args.output)}")


if __name__ == "__main__":
    main()
