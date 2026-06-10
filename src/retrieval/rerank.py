from __future__ import annotations


def passthrough_rerank(hits: list, top_k: int | None = None) -> list:
    """Placeholder reranker. Keeps retrieval deterministic until a cross-encoder is added."""
    return hits[:top_k] if top_k else hits
