from __future__ import annotations

import argparse
from pathlib import Path

import faiss

from src.common import add_config_arg, ensure_dir, load_config, project_path, read_jsonl, resolve_model_ref, write_jsonl
from src.models.embeddings import EmbeddingModel


def build_index(config: dict) -> tuple[faiss.Index, list[dict]]:
    processed_dir = Path(config["paths"]["processed_dir"])
    chunks_path = processed_dir / "rule_chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise FileNotFoundError(f"No chunks found at {project_path(chunks_path)}. Run build_rule_corpus first.")

    model = EmbeddingModel(
        resolve_model_ref(config, "embedding"),
        cache_dir=config["paths"].get("model_cache_dir"),
    )
    texts = [row["retrieval_text"] for row in chunks]
    vectors = model.encode(texts, normalize=config["retrieval"].get("normalize_embeddings", True))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    args = parser.parse_args()
    config = load_config(args.config)

    index, chunks = build_index(config)
    index_path = project_path(config["paths"]["index_path"])
    ensure_dir(index_path.parent)
    faiss.write_index(index, str(index_path))
    write_jsonl(config["paths"]["metadata_path"], chunks)
    print(f"Wrote FAISS index with {index.ntotal} vectors to {index_path}")
    print(f"Wrote metadata to {project_path(config['paths']['metadata_path'])}")


if __name__ == "__main__":
    main()
