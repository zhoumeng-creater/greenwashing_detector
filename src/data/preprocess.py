from __future__ import annotations

import re
from dataclasses import dataclass


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_id: str
    title: str
    jurisdiction: str
    claim_type: str
    text: str
    url: str


def chunk_text(
    *,
    doc_id: str,
    source_id: str,
    title: str,
    jurisdiction: str,
    claim_type: str,
    text: str,
    url: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        snippet = text[start:end].strip()
        if snippet:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_C{index:04d}",
                    doc_id=doc_id,
                    source_id=source_id,
                    title=title,
                    jurisdiction=jurisdiction,
                    claim_type=claim_type,
                    text=snippet,
                    url=url,
                )
            )
        if end >= len(text):
            break
        start = max(end - chunk_overlap_chars, start + 1)
        index += 1
    return chunks
