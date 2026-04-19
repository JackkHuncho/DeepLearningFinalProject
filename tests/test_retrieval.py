"""Tests for the retrieval layer: index, retriever, ingest, and schemas."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from f1_commentary.config import RetrievalConfig
from f1_commentary.retrieval.index import (
    NumpyIndex,
    VectorIndex,
    create_index,
)
from f1_commentary.retrieval.ingest import IndexBuilder
from f1_commentary.retrieval.retriever import Retriever
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    DriverState,
    EventType,
    RaceStateSnapshot,
)
from f1_commentary.schemas.memory import MemoryChunk


# ---------------------------------------------------------------------------
# Mock embedding model (no sentence_transformers dependency)
# ---------------------------------------------------------------------------


class MockEmbedModel:
    def encode(self, texts):
        rng = np.random.default_rng(42)
        embs = rng.random((len(texts), 64)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    def encode_single(self, text):
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(n: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def _make_chunks(n: int, drivers: list[list[str]] | None = None) -> list[MemoryChunk]:
    chunks = []
    for i in range(n):
        d = drivers[i] if drivers else [f"DR{i}"]
        chunks.append(
            MemoryChunk(
                chunk_id=f"chunk_{i}",
                year=2024,
                grand_prix="Monza",
                session="R",
                lap_window=(1, 5),
                event_type="overtake_opportunity",
                involved_drivers=d,
                text=f"Sample text for chunk {i}",
            )
        )
    return chunks


def _make_state() -> RaceStateSnapshot:
    return RaceStateSnapshot(
        timestamp=datetime.utcnow(),
        event_type=EventType.RACE_STATE_UPDATE,
        session_key="2024_Monza_R",
        lap=10,
        drivers=[
            DriverState(driver_id="HAM", position=1, lap_number=10),
            DriverState(driver_id="VER", position=2, lap_number=10),
        ],
    )


def _make_candidate(
    confidence: float = 0.9,
    drivers: list[str] | None = None,
    explanation: str = "battle for position",
) -> CandidateEvent:
    return CandidateEvent(
        timestamp=datetime.utcnow(),
        event_type=EventType.CANDIDATE_EVENT,
        candidate_type=CandidateEventType.OVERTAKE_OPPORTUNITY,
        involved_drivers=drivers or ["HAM", "VER"],
        confidence=confidence,
        explanation=explanation,
        lap=10,
        session_key="2024_Monza_R",
    )


# ===========================================================================
# 1. NumpyIndex add + search
# ===========================================================================


class TestNumpyIndex:
    def test_add_and_search(self):
        idx = NumpyIndex()
        embs = _make_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        idx.add(embs, ids)

        assert idx.size == 10

        results = idx.search(embs[0], k=3)
        assert len(results) == 3
        # First result should be the query vector itself (highest similarity)
        assert results[0][0] == "id_0"
        # Scores should be in descending order
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    # 2. NumpyIndex save/load round-trip
    def test_save_load_roundtrip(self):
        idx = NumpyIndex()
        embs = _make_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        idx.add(embs, ids)

        original_results = idx.search(embs[0], k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"
            idx.save(save_path)
            loaded = NumpyIndex.load(save_path)

        assert loaded.size == 10
        loaded_results = loaded.search(embs[0], k=3)
        assert [r[0] for r in original_results] == [r[0] for r in loaded_results]
        for (_, s1), (_, s2) in zip(original_results, loaded_results):
            assert abs(s1 - s2) < 1e-5

    # 3. NumpyIndex empty
    def test_empty_search(self):
        idx = NumpyIndex()
        query = _make_embeddings(1)[0]
        results = idx.search(query, k=5)
        assert results == []
        assert idx.size == 0


# ===========================================================================
# 4. FaissIndex add + search
# ===========================================================================


class TestFaissIndex:
    def test_add_and_search(self):
        faiss = pytest.importorskip("faiss")
        from f1_commentary.retrieval.index import FaissIndex

        idx = FaissIndex()
        embs = _make_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        idx.add(embs, ids)

        assert idx.size == 10

        results = idx.search(embs[0], k=3)
        assert len(results) == 3
        # First result should be the query vector itself
        assert results[0][0] == "id_0"
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    # 5. FaissIndex save/load round-trip
    def test_save_load_roundtrip(self):
        faiss = pytest.importorskip("faiss")
        from f1_commentary.retrieval.index import FaissIndex

        idx = FaissIndex()
        embs = _make_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        idx.add(embs, ids)

        original_results = idx.search(embs[0], k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"
            idx.save(save_path)
            loaded = FaissIndex.load(save_path)

        assert loaded.size == 10
        loaded_results = loaded.search(embs[0], k=3)
        assert [r[0] for r in original_results] == [r[0] for r in loaded_results]


# ===========================================================================
# 6. create_index factory
# ===========================================================================


def test_create_index_factory():
    idx = create_index(use_faiss=False)
    assert isinstance(idx, VectorIndex)
    assert isinstance(idx, NumpyIndex)


# ===========================================================================
# 7. MemoryChunk schema
# ===========================================================================


def test_memory_chunk_schema():
    chunk = MemoryChunk(
        chunk_id="test_chunk",
        year=2024,
        grand_prix="Monza",
        session="R",
        lap_window=(1, 5),
        event_type="overtake_opportunity",
        involved_drivers=["HAM", "VER"],
        text="Hamilton overtakes Verstappen on lap 3",
    )
    assert chunk.chunk_id == "test_chunk"
    assert chunk.year == 2024
    assert chunk.involved_drivers == ["HAM", "VER"]

    # Serialize to JSON and back
    json_str = chunk.model_dump_json()
    data = json.loads(json_str)
    assert data["chunk_id"] == "test_chunk"
    assert data["text"] == "Hamilton overtakes Verstappen on lap 3"
    roundtripped = MemoryChunk(**data)
    assert roundtripped.chunk_id == chunk.chunk_id


# ===========================================================================
# 8. Retriever tier1: filter by involved_drivers
# ===========================================================================


class TestRetriever:
    def _build_retriever(
        self, chunks: list[MemoryChunk], config: RetrievalConfig | None = None
    ) -> Retriever:
        config = config or RetrievalConfig(use_faiss=False)
        model = MockEmbedModel()
        # Build an index from chunks
        builder = IndexBuilder(model, config)
        index, _ = builder.build_from_chunks(chunks)
        return Retriever(
            index=index,
            chunks=chunks,
            embed_fn=model.encode_single,
            config=config,
        )

    def test_tier1_metadata_filter(self):
        drivers_list = [["HAM", "VER"], ["LEC", "SAI"], ["HAM", "NOR"]]
        chunks = _make_chunks(3, drivers=drivers_list)
        retriever = self._build_retriever(chunks)
        state = _make_state()

        # Event involves HAM — should match chunks 0 and 2
        event = _make_candidate(confidence=0.5, drivers=["HAM"])
        tier1 = retriever.tier1_metadata(event, state)
        matched_ids = {c.chunk_id for c in tier1}
        assert "chunk_0" in matched_ids
        assert "chunk_2" in matched_ids
        assert "chunk_1" not in matched_ids

    # 9. Retriever tier2: semantic search returns ranked chunks
    def test_tier2_semantic(self):
        chunks = _make_chunks(10)
        retriever = self._build_retriever(chunks)
        results = retriever.tier2_semantic("overtake battle", k=5)
        assert len(results) == 5
        # Results are (MemoryChunk, score) tuples
        for chunk, score in results:
            assert isinstance(chunk, MemoryChunk)
            assert isinstance(score, float)
        # Scores should be in descending order
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    # 10. Retriever gating: low confidence -> gated_out=True, tier=1
    def test_gating_low_confidence(self):
        chunks = _make_chunks(5)
        config = RetrievalConfig(use_faiss=False, tier2_priority_threshold=0.7)
        retriever = self._build_retriever(chunks, config)
        state = _make_state()

        event = _make_candidate(confidence=0.3, drivers=["DR0"])
        result = retriever.retrieve_context_for_event(event, state)
        assert result.gated_out is True
        assert result.tier == 1

    # 11. Retriever gating: high confidence -> gated_out=False, tier=2
    def test_gating_high_confidence(self):
        chunks = _make_chunks(5)
        config = RetrievalConfig(use_faiss=False, tier2_priority_threshold=0.7)
        retriever = self._build_retriever(chunks, config)
        state = _make_state()

        event = _make_candidate(confidence=0.9, drivers=["DR0"])
        result = retriever.retrieve_context_for_event(event, state)
        assert result.gated_out is False
        assert result.tier == 2


# ===========================================================================
# 12. IndexBuilder build_from_chunks
# ===========================================================================


def test_index_builder_build_from_chunks():
    model = MockEmbedModel()
    config = RetrievalConfig(use_faiss=False)
    builder = IndexBuilder(model, config)
    chunks = _make_chunks(8)

    index, returned_chunks = builder.build_from_chunks(chunks)
    assert index.size == 8
    assert len(returned_chunks) == 8
    assert isinstance(index, VectorIndex)
