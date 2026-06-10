from __future__ import annotations

import re
from typing import Any


LEGAL_FINALITY_PATTERNS = [
    r"\billegal\b",
    r"\bfraud\b",
    r"\bfraudulent\b",
    r"\bdeceptive by law\b",
    r"\bviolates the law\b",
]

PII_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
]


def apply_ethics_rubric(record: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(record.get(key, ""))
        for key in ["reasoning_summary", "responsible_rewrite", "consumer_explanation"]
    ).lower()
    legal_finality = [pattern for pattern in LEGAL_FINALITY_PATTERNS if re.search(pattern, text)]
    pii = [pattern for pattern in PII_PATTERNS if re.search(pattern, text)]
    evidence_insufficient = record.get("greenwashing_risk_level") == "INSUFFICIENT_EVIDENCE"
    high_risk_without_citations = (
        record.get("greenwashing_risk_level") == "HIGH"
        and not record.get("citations")
    )
    issues = []
    if legal_finality:
        issues.append("Avoid final legal accusations; use risk-indication language.")
    if pii:
        issues.append("Remove or generalize personal information.")
    if high_risk_without_citations:
        issues.append("High risk requires cited evidence or guidance.")
    if evidence_insufficient:
        issues.append("Evidence gap should be clearly disclosed.")
    return {
        "legal_finality_patterns": legal_finality,
        "pii_patterns": pii,
        "high_risk_without_citations": high_risk_without_citations,
        "issues": issues,
        "passes": not legal_finality and not pii and not high_risk_without_citations,
    }
