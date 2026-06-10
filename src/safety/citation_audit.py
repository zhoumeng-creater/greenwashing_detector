from __future__ import annotations

from typing import Any


def citation_audit(record: dict[str, Any], retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    available_ids = {row.get("chunk_id") for row in retrieved_chunks} | {row.get("source_id") for row in retrieved_chunks}
    citations = set(record.get("citations", []))
    missing = sorted(cite for cite in citations if cite not in available_ids)
    risk_categories = record.get("risk_categories", [])
    evidence_table = record.get("evidence_table", [])
    has_evidence = bool(evidence_table) or bool(citations)
    return {
        "available_source_count": len(available_ids),
        "citation_count": len(citations),
        "missing_citations": missing,
        "risk_categories_have_evidence": not risk_categories or has_evidence,
        "passes": not missing and (not risk_categories or has_evidence),
    }
