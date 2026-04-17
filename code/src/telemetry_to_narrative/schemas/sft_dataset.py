"""Dataset schemas for the SFT supervision pipeline.

These schemas define the full journey from raw transcript segments through
event windows to final SFT-ready training examples.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo
from src.telemetry_to_narrative.schemas.candidate_event import EventType


# ── Transcript ────────────────────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    """One segment of commentary transcript with a timestamp."""

    segment_id: str = ""
    timestamp: datetime
    end_timestamp: Optional[datetime] = None
    speaker: Optional[str] = None
    text: str
    drivers_mentioned: list[str] = Field(default_factory=list)
    event_type_hint: Optional[str] = Field(
        default=None,
        description="Optional coarse event category from transcript metadata",
    )


# ── Event Window ──────────────────────────────────────────────────────────

class EventWindow(BaseModel):
    """A local context window around a selected beat.

    Groups the snapshot state and beat metadata needed to construct
    one SFT training example.
    """

    window_id: str
    frame_ids: list[int] = Field(
        description="Frame IDs in this window (centre ± context)"
    )
    timestamps: list[datetime] = Field(
        description="Corresponding timestamps for each frame in the window"
    )
    centre_frame_id: int
    centre_timestamp: datetime
    session_info: SessionInfo
    event_type: EventType
    storyline_id: Optional[str] = None
    involved_drivers: list[str] = Field(default_factory=list)
    priority_score: float = 0.0
    beat_description: Optional[str] = None
    state_summary: dict = Field(
        default_factory=dict,
        description="Compact snapshot state at the centre frame",
    )

    model_config = {"ser_json_timedelta": "float"}


# ── Alignment ─────────────────────────────────────────────────────────────

class AlignmentConfidence(str, Enum):
    """How confidently a transcript was matched to an event window."""

    EXACT = "exact"        # timestamp falls inside the window
    NEAR = "near"          # within the configurable tolerance
    HEURISTIC = "heuristic"  # matched by driver/event-type hints only
    UNMATCHED = "unmatched"  # no alignment found


# ── SFT Example ───────────────────────────────────────────────────────────

class SourceRef(BaseModel):
    """Provenance pointer back to raw data."""

    snapshot_frame_id: int
    beat_id: Optional[str] = None
    transcript_segment_id: Optional[str] = None


class SFTExample(BaseModel):
    """One supervised fine-tuning example.

    Contains both structured fields (for analysis / filtering) and
    the final prompt-ready ``input_text`` / ``target_text`` pair.
    """

    example_id: str
    session_info: SessionInfo
    event_type: EventType
    storyline_id: Optional[str] = None
    involved_drivers: list[str] = Field(default_factory=list)

    # Structured input components.
    observed_facts: list[str] = Field(
        default_factory=list,
        description="Direct state facts (e.g. 'VER is P1 on lap 18')",
    )
    derived_inferences: list[str] = Field(
        default_factory=list,
        description="Computed inferences (e.g. 'overtake threat increasing')",
    )
    retrieved_context: list[str] = Field(
        default_factory=list,
        description="Placeholder for Phase 6 retrieved context",
    )

    # Final formatted strings.
    input_text: str = Field(
        description="Formatted prompt-ready input for the model"
    )
    target_text: str = Field(
        description="Clean commentary text (training target)"
    )

    # Metadata.
    alignment_confidence: AlignmentConfidence = AlignmentConfidence.UNMATCHED
    source_refs: list[SourceRef] = Field(default_factory=list)

    model_config = {"ser_json_timedelta": "float"}


# ── Dataset Metadata ─────────────────────────────────────────────────────

class DatasetMetadata(BaseModel):
    """Metadata for a generated SFT dataset."""

    created_at: datetime
    pipeline_version: str = "0.1.0"
    total_examples: int = 0
    train_examples: int = 0
    val_examples: int = 0
    sessions: list[dict] = Field(
        default_factory=list,
        description="List of {year, grand_prix, session_type} included",
    )
    event_type_distribution: dict[str, int] = Field(default_factory=dict)
    alignment_distribution: dict[str, int] = Field(default_factory=dict)
    quality_report: dict = Field(
        default_factory=dict,
        description="Summary of quality checks run",
    )
    split_seed: int = 42
    val_fraction: float = 0.2
    transcript_sources: list[str] = Field(default_factory=list)
