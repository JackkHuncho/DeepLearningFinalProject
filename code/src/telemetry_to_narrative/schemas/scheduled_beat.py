"""ScheduledBeat schema — the output of the EditorialScheduler.

Each beat represents one narration decision: whether to generate
commentary for a candidate event, suppress it, delay it, or merge it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.candidate_event import EventType


class SuppressionReason(str, Enum):
    """Why a beat was not selected for generation."""

    NONE = "none"                             # beat was selected
    LOW_PRIORITY = "low_priority"             # score below narration threshold
    STORYLINE_NOT_ADVANCED = "storyline_not_advanced"  # same storyline, no change
    BACKPRESSURE_DROP = "backpressure_drop"   # system load too high
    DUPLICATE_SEMANTICS = "duplicate_semantics"  # same info already narrated
    COOLDOWN = "cooldown"                     # storyline narrated too recently


class ScoreBreakdown(BaseModel):
    """Decomposed priority score for transparency and paper reporting.

    Each component is in [0, 1] before weighting; the final priority_score
    is their weighted sum, possibly adjusted by backpressure.
    """

    contextual_importance: float = Field(
        default=0.0,
        description="Based on event type editorial precedence tier",
    )
    urgency: float = Field(
        default=0.0,
        description="Based on gap, intensity, and time-sensitivity",
    )
    race_phase: float = Field(
        default=0.0,
        description="Boost for events in critical race phases (start, end, restarts)",
    )
    storyline_relevance: float = Field(
        default=0.0,
        description="Higher if this event continues an active storyline",
    )
    confidence: float = Field(
        default=1.0,
        description="Event detector confidence (pass-through from CandidateEvent)",
    )
    backpressure_penalty: float = Field(
        default=0.0,
        description="Penalty applied when system is under load (subtracted)",
    )


class ScheduledBeat(BaseModel):
    """One scheduling decision for a candidate event."""

    beat_id: str
    timestamp: datetime
    event_type: EventType
    involved_drivers: list[str] = Field(default_factory=list)
    storyline_id: Optional[str] = Field(
        default=None,
        description="Storyline this beat belongs to (e.g. 'battle_VER_HAM')",
    )
    priority_score: float = Field(
        default=0.0,
        description="Final composite priority score after all adjustments",
    )
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    selected_for_generation: bool = Field(
        default=False,
        description="True if this beat should be narrated",
    )
    suppression_reason: SuppressionReason = Field(
        default=SuppressionReason.NONE,
    )
    retrieval_allowed: bool = Field(
        default=True,
        description="False if backpressure disabled retrieval for this beat",
    )
    output_length_budget: int = Field(
        default=150,
        description="Target word count for generated commentary",
    )
    source_frame_id: int = Field(default=0, ge=0)
    description: Optional[str] = None

    model_config = {"ser_json_timedelta": "float"}
