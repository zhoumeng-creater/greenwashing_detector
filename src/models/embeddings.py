from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common import project_path


class EmbeddingModel:
    def __init__(self, model_name: str, cache_dir: str | None = None, device: str | None = None):
        kwargs = {}
        if cache_dir:
            kwargs["cache_folder"] = str(project_path(cache_dir))
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Iterable[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype("float32")
