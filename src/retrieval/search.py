from __future__ import annotations

import argparse
from dataclasses import dataclass

import faiss

from src.common import add_config_arg, load_config, project_path, read_jsonl, resolve_model_ref
from src.models.embeddings import EmbeddingModel


@dataclass
class RetrievalHit:
    score: float
    chunk: dict


class FaissRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.index = faiss.read_index(str(project_path(config["paths"]["index_path"])))
        self.metadata = read_jsonl(config["paths"]["metadata_path"])
        self.embedding_model = EmbeddingModel(
            resolve_model_ref(config, "embedding"),
            cache_dir=config["paths"].get("model_cache_dir"),
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.config["retrieval"]["top_k"]
        vector = self.embedding_model.encode([query], normalize=self.config["retrieval"].get("normalize_embeddings", True))
        scores, indices = self.index.search(vector, top_k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(RetrievalHit(float(score), self.metadata[int(idx)]))
        return hits


def build_query(claim: str, product_category: str = "", jurisdiction: str = "") -> str:
    parts = [
        "environmental marketing claim greenwashing guidance",
        claim,
    ]
    if product_category:
        parts.append(f"product category: {product_category}")
    if jurisdiction:
        parts.append(f"jurisdiction: {jurisdiction}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    retriever = FaissRetriever(config)
    for hit in retriever.search(args.query, top_k=args.top_k):
        chunk = hit.chunk
        print(f"[{hit.score:.4f}] {chunk['chunk_id']} {chunk['title']} ({chunk['source_id']})")
        print(chunk["text"][:500].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
