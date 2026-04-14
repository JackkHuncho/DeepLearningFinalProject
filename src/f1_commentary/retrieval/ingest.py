"""Index builder: embed chunks and persist a VectorIndex to disk.

Reads ``MemoryChunk`` objects, encodes their text with an embedding
model, builds a ``VectorIndex``, and saves both the index and the
chunk metadata for later use by the ``Retriever``.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from f1_commentary.config import RetrievalConfig
from f1_commentary.retrieval.index import VectorIndex, create_index
from f1_commentary.schemas.memory import MemoryChunk


class EmbedModel(Protocol):
    """Minimal embedding-model interface used by IndexBuilder."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class IndexBuilder:
    """Builds a VectorIndex from a list of MemoryChunks."""

    def __init__(self, embed_model: EmbedModel, config: RetrievalConfig) -> None:
        self._embed_model = embed_model
        self._config = config

    def build_from_chunks(
        self, chunks: list[MemoryChunk]
    ) -> tuple[VectorIndex, list[MemoryChunk]]:
        """Embed all chunk texts and build a vector index.

        Returns ``(index, chunks)`` where the index IDs correspond to
        ``chunk.chunk_id`` values.
        """
        texts = [c.text for c in chunks]
        embeddings = self._embed_model.encode(texts)

        index = create_index(use_faiss=self._config.use_faiss)
        ids = [c.chunk_id for c in chunks]
        index.add(embeddings, ids)
        return index, chunks

    def save(
        self,
        index: VectorIndex,
        chunks: list[MemoryChunk],
        output_dir: Path,
    ) -> None:
        """Save the index and chunk metadata to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vector index
        index.save(output_dir / "index")

        # Save chunk metadata as JSON
        chunk_dicts = [c.model_dump(mode="json") for c in chunks]
        with open(output_dir / "chunks.json", "w") as f:
            json.dump(chunk_dicts, f, indent=2, default=str)

    @staticmethod
    def load(
        output_dir: Path, config: RetrievalConfig
    ) -> tuple[VectorIndex, list[MemoryChunk]]:
        """Load a previously-saved index and its chunks."""
        output_dir = Path(output_dir)

        # Determine which index type was saved
        index_dir = output_dir / "index"
        if (index_dir / "faiss.index").exists():
            from f1_commentary.retrieval.index import FaissIndex

            index = FaissIndex.load(index_dir)
        else:
            from f1_commentary.retrieval.index import NumpyIndex

            index = NumpyIndex.load(index_dir)

        # Load chunks
        with open(output_dir / "chunks.json") as f:
            chunk_dicts = json.load(f)
        chunks = [MemoryChunk(**d) for d in chunk_dicts]

        return index, chunks
