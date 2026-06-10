from __future__ import annotations

from src.retrieval.search import RetrievalHit


def format_context(hits: list[RetrievalHit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk
        source_label = chunk.get("chunk_id", f"D{i:02d}")
        blocks.append(
            "\n".join(
                [
                    f"[{source_label}]",
                    f"Title: {chunk.get('title', '')}",
                    f"Source: {chunk.get('source_id', '')}",
                    f"Jurisdiction: {chunk.get('jurisdiction', '')}",
                    f"Claim type: {chunk.get('claim_type', '')}",
                    f"URL: {chunk.get('url', '')}",
                    f"Text: {chunk.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_analysis_prompt(
    *,
    claim: str,
    product_category: str,
    jurisdiction: str,
    evidence_text: str,
    context: str,
) -> str:
    return f"""You are an environmental marketing claim compliance assistant.
Use only the retrieved guidelines and evidence.
Do not accuse a company of intentional deception unless explicit evidence supports it.

Claim:
{claim}

Product/category:
{product_category or "unknown"}

Target jurisdiction:
{jurisdiction or "general"}

Optional product evidence supplied by user:
{evidence_text or "None"}

Retrieved guidelines and evidence:
{context}

Return valid JSON with exactly these keys:
- is_environmental_claim: true or false
- greenwashing_risk_level: LOW, MEDIUM, HIGH, or INSUFFICIENT_EVIDENCE
- risk_score: number from 0 to 1
- risk_categories: list using only:
  vague_general_claim, unsubstantiated_claim, absolute_claim, hidden_tradeoff,
  misleading_comparison, recyclability_ambiguity, carbon_offset_or_net_zero_risk,
  certification_ambiguity, future_target_overclaim, nature_imagery_mismatch
- evidence_table: list of objects with source_id, source_type, relevance, summary
- reasoning_summary: concise explanation grounded in citations
- responsible_rewrite: a more specific, qualified, evidence-aware rewrite
- consumer_explanation: plain-language consumer-facing explanation
- citations: list of cited source IDs
- ethics_notes: list of caution notes

Rules:
1. If evidence is insufficient, say evidence is insufficient.
2. Do not overstate legal conclusions.
3. Every risk category must cite at least one guideline or evidence source.
4. Prefer specific, qualified rewrites over broad green language.
5. Return JSON only; do not include markdown.
"""
