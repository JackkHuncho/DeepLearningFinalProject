"""Two-tier retriever: fast metadata filtering + semantic search.

Tier 1 always runs (metadata match on drivers, event type, session).
Tier 2 (semantic / embedding search) is gated by event confidence.
"""

import time
from datetime import datetime, timezone
from typing import Callable, Protocol

import numpy as np

from f1_commentary.config import RetrievalConfig
from f1_commentary.retrieval.index import VectorIndex
from f1_commentary.schemas.events import (
    CandidateEvent,
    EventType,
    RaceStateSnapshot,
    RetrievalResult,
)
from f1_commentary.schemas.memory import MemoryChunk


class EmbedFn(Protocol):
    """Callable that encodes a single text into a numpy vector."""

    def __call__(self, text: str) -> np.ndarray: ...


class Retriever:
    """Two-tier retrieval engine."""

    def __init__(
        self,
        index: VectorIndex,
        chunks: list[MemoryChunk],
        embed_fn: Callable[[str], np.ndarray],
        config: RetrievalConfig,
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._chunk_map: dict[str, MemoryChunk] = {
            c.chunk_id: c for c in chunks
        }
        self._embed_fn = embed_fn
        self._config = config

    # -- Tier 1: metadata filtering ------------------------------------------

    def tier1_metadata(
        self, event: CandidateEvent, state: RaceStateSnapshot
    ) -> list[MemoryChunk]:
        """Fast metadata filtering: match by drivers, event type, session."""
        event_drivers = set(event.involved_drivers)
        results: list[MemoryChunk] = []
        for chunk in self._chunks:
            # Match by involved drivers (any overlap)
            if event_drivers and chunk.involved_drivers:
                if not event_drivers.intersection(chunk.involved_drivers):
                    continue
            elif event_drivers and not chunk.involved_drivers:
                # Chunk has no driver info — still include as general context
                pass

            # Match by event type if available
            if (
                chunk.event_type is not None
                and event.candidate_type is not None
                and chunk.event_type != event.candidate_type.value
            ):
                continue

            results.append(chunk)
        return results

    # -- Tier 2: semantic search ---------------------------------------------

    def tier2_semantic(
        self, query_text: str, k: int = 5
    ) -> list[tuple[MemoryChunk, float]]:
        """Semantic search using embeddings."""
        query_vec = self._embed_fn(query_text)
        hits = self._index.search(query_vec, k=k)
        results: list[tuple[MemoryChunk, float]] = []
        for chunk_id, score in hits:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is not None:
                results.append((chunk, score))
        return results

    # -- Combined gated retrieval --------------------------------------------

    def retrieve_context_for_event(
        self, event: CandidateEvent, state: RaceStateSnapshot
    ) -> RetrievalResult:
        """Gated retrieval: Tier 1 always, Tier 2 only if confidence >= threshold."""
        now = datetime.now(tz=timezone.utc)

        # Tier 1 — always
        t1_start = time.perf_counter()
        tier1_chunks = self.tier1_metadata(event, state)
        t1_ms = (time.perf_counter() - t1_start) * 1000

        # Decide whether to run Tier 2
        run_tier2 = event.confidence >= self._config.tier2_priority_threshold

        if not run_tier2:
            return RetrievalResult(
                timestamp=now,
                event_type=EventType.RETRIEVAL_RESULT,
                query_event_id=event.event_id,
                tier=1,
                chunks=[c.model_dump(mode="json") for c in tier1_chunks],
                latency_ms=t1_ms,
                gated_out=True,
            )

        # Tier 2 — semantic
        query_text = event.explanation or " ".join(event.involved_drivers)
        t2_start = time.perf_counter()
        tier2_results = self.tier2_semantic(query_text, k=5)
        t2_ms = (time.perf_counter() - t2_start) * 1000

        # Merge: tier1 metadata chunks + tier2 semantic ranked results
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for chunk, score in tier2_results:
            if chunk.chunk_id not in seen_ids:
                d = chunk.model_dump(mode="json")
                d["semantic_score"] = score
                merged.append(d)
                seen_ids.add(chunk.chunk_id)
        for chunk in tier1_chunks:
            if chunk.chunk_id not in seen_ids:
                merged.append(chunk.model_dump(mode="json"))
                seen_ids.add(chunk.chunk_id)

        total_ms = t1_ms + t2_ms
        return RetrievalResult(
            timestamp=now,
            event_type=EventType.RETRIEVAL_RESULT,
            query_event_id=event.event_id,
            tier=2,
            chunks=merged,
            latency_ms=total_ms,
            gated_out=False,
        )
