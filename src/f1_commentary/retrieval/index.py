"""Vector index abstraction with FAISS and pure-numpy backends.

Provides a common ``VectorIndex`` interface with two concrete
implementations: ``FaissIndex`` (fast, requires ``faiss-cpu``) and
``NumpyIndex`` (pure-numpy cosine-similarity fallback).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json

import numpy as np


class VectorIndex(ABC):
    """Abstract base class for a vector similarity index."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add *embeddings* (N x D) with corresponding *ids*."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return up to *k* ``(id, score)`` pairs sorted by descending similarity."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the index to *path* (may create multiple files)."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Restore a previously-saved index from *path*."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently stored."""
        ...


# ---------------------------------------------------------------------------
# Pure-numpy fallback
# ---------------------------------------------------------------------------


class NumpyIndex(VectorIndex):
    """Pure-numpy cosine-similarity fallback. Always works."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._ids: list[str] = []

    # -- VectorIndex interface ------------------------------------------------

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        if embeddings.shape[0] != len(ids):
            raise ValueError("Number of embeddings must match number of ids")
        # Normalize to unit length for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms

        if self._embeddings is None:
            self._embeddings = normed.astype(np.float32)
        else:
            self._embeddings = np.vstack(
                [self._embeddings, normed.astype(np.float32)]
            )
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self._embeddings is None or len(self._ids) == 0:
            return []
        # Normalize query
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            return []
        q = (query / qnorm).astype(np.float32)

        scores = self._embeddings @ q  # cosine similarities
        k = min(k, len(self._ids))
        if k >= len(scores):
            # No need to partition — just sort everything
            top_k_idx = np.argsort(-scores)[:k]
        else:
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [(self._ids[i], float(scores[i])) for i in top_k_idx]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
        with open(path / "ids.json", "w") as f:
            json.dump(self._ids, f)

    @classmethod
    def load(cls, path: Path) -> "NumpyIndex":
        path = Path(path)
        idx = cls()
        emb_path = path / "embeddings.npy"
        ids_path = path / "ids.json"
        if emb_path.exists():
            idx._embeddings = np.load(emb_path)
        with open(ids_path) as f:
            idx._ids = json.load(f)
        return idx

    @property
    def size(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# FAISS wrapper
# ---------------------------------------------------------------------------


class FaissIndex(VectorIndex):
    """FAISS ``IndexFlatIP`` wrapper. Faster for larger indices.

    Uses inner-product on L2-normalised vectors, which is equivalent to
    cosine similarity.
    """

    def __init__(self, dim: int | None = None) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FaissIndex. "
                "Install it with: pip install faiss-cpu"
            )
        self._faiss = faiss
        self._dim = dim
        self._index: object | None = None  # faiss.IndexFlatIP
        self._ids: list[str] = []

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = self._faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        if embeddings.shape[0] != len(ids):
            raise ValueError("Number of embeddings must match number of ids")
        # Normalize
        emb = embeddings.astype(np.float32)
        self._faiss.normalize_L2(emb)

        self._ensure_index(emb.shape[1])
        self._index.add(emb)  # type: ignore[union-attr]
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self._index is None or len(self._ids) == 0:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        self._faiss.normalize_L2(q)
        k = min(k, len(self._ids))
        scores, indices = self._index.search(q, k)  # type: ignore[union-attr]
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "ids.json", "w") as f:
            json.dump(self._ids, f)

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        import faiss as _faiss

        path = Path(path)
        idx = cls.__new__(cls)
        idx._faiss = _faiss
        idx._ids = []
        idx._index = None
        idx._dim = None

        index_path = path / "faiss.index"
        ids_path = path / "ids.json"
        if index_path.exists():
            idx._index = _faiss.read_index(str(index_path))
            idx._dim = idx._index.d
        with open(ids_path) as f:
            idx._ids = json.load(f)
        return idx

    @property
    def size(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_index(use_faiss: bool = True) -> VectorIndex:
    """Factory: try FAISS first, fall back to NumpyIndex."""
    if use_faiss:
        try:
            return FaissIndex()
        except ImportError:
            pass
    return NumpyIndex()
