"""Memory record schemas for the retrieval subsystem.

Defines the document and chunk models stored in the vector index,
including metadata for provenance tracking.
"""

from datetime import datetime, timezone
from typing import Optional

import uuid

from pydantic import BaseModel, Field


class MemoryChunk(BaseModel):
    """A single chunk of contextual memory stored in the retrieval index."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    year: int
    grand_prix: str
    session: str  # "FP1", "FP2", "FP3", "Q", "R"
    lap_window: tuple[int, int]  # (start_lap, end_lap)
    event_type: Optional[str] = None
    involved_drivers: list[str] = Field(default_factory=list)
    text: str  # The actual content
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
