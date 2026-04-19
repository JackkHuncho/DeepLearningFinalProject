"""Embedding helper wrapping sentence-transformers.

Provides a thin interface around ``SentenceTransformer`` so that the
rest of the retrieval pipeline only depends on ``encode`` / ``encode_single``.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalised embeddings."""
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]
