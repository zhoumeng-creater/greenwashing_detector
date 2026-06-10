from src.data.build_rule_corpus import build_rule_chunks
from src.common import load_config


def test_seed_rule_chunks_build_without_downloaded_pages():
    config = load_config("configs/default.yaml")
    chunks = build_rule_chunks(config, include_downloaded=False)
    assert len(chunks) >= 5
    assert all("retrieval_text" in chunk for chunk in chunks)
