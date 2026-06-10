from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator


RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "INSUFFICIENT_EVIDENCE"]

RISK_CATEGORIES = [
    "vague_general_claim",
    "unsubstantiated_claim",
    "absolute_claim",
    "hidden_tradeoff",
    "misleading_comparison",
    "recyclability_ambiguity",
    "carbon_offset_or_net_zero_risk",
    "certification_ambiguity",
    "future_target_overclaim",
    "nature_imagery_mismatch",
]

GREENWASHING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "is_environmental_claim",
        "greenwashing_risk_level",
        "risk_score",
        "risk_categories",
        "evidence_table",
        "reasoning_summary",
        "responsible_rewrite",
        "consumer_explanation",
        "citations",
        "ethics_notes",
    ],
    "properties": {
        "is_environmental_claim": {"type": "boolean"},
        "greenwashing_risk_level": {"type": "string", "enum": RISK_LEVELS},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "risk_categories": {"type": "array", "items": {"type": "string", "enum": RISK_CATEGORIES}},
        "evidence_table": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_id", "source_type", "relevance", "summary"],
                "properties": {
                    "source_id": {"type": "string"},
                    "source_type": {"type": "string"},
                    "relevance": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        },
        "reasoning_summary": {"type": "string"},
        "responsible_rewrite": {"type": "string"},
        "consumer_explanation": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "ethics_notes": {"type": "array", "items": {"type": "string"}},
    },
}

VALIDATOR = Draft202012Validator(GREENWASHING_SCHEMA)


def validate_output(record: dict[str, Any]) -> list[str]:
    return [error.message for error in VALIDATOR.iter_errors(record)]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in generated text.")
    return json.loads(text[start : end + 1])


def fallback_output(claim: str, reason: str) -> dict[str, Any]:
    return {
        "is_environmental_claim": True,
        "greenwashing_risk_level": "INSUFFICIENT_EVIDENCE",
        "risk_score": 0.0,
        "risk_categories": [],
        "evidence_table": [],
        "reasoning_summary": f"Unable to produce a validated analysis for the claim: {claim}. Reason: {reason}",
        "responsible_rewrite": "Evidence is insufficient. Provide specific environmental attributes, scope, and substantiation before making a claim.",
        "consumer_explanation": "The system could not verify the claim with available evidence.",
        "citations": [],
        "ethics_notes": ["Do not make a strong accusation without evidence."],
    }
