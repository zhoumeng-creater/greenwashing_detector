from __future__ import annotations

import argparse

import faiss

from src.common import add_config_arg, load_config, project_path, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    args = parser.parse_args()
    config = load_config(args.config)

    index_path = project_path(config["paths"]["index_path"])
    metadata = read_jsonl(config["paths"]["metadata_path"])
    index = faiss.read_index(str(index_path))
    print(f"Index path: {index_path}")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    print(f"Metadata rows: {len(metadata)}")
    if metadata:
        first = metadata[0]
        print(f"First chunk: {first['chunk_id']} | {first['title']}")


if __name__ == "__main__":
    main()
