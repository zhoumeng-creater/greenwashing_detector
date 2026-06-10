from __future__ import annotations


VAGUE_TERMS = [
    "eco-friendly",
    "green",
    "sustainable",
    "planet-friendly",
    "environmentally friendly",
    "100%",
    "zero impact",
    "completely recyclable",
]


def count_vague_terms(text: str) -> int:
    lower = text.lower()
    return sum(1 for term in VAGUE_TERMS if term in lower)
