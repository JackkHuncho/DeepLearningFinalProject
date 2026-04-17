"""FinalCommentary schema — the last stage of the narrative pipeline.

After generation, the BeatManager post-processes commentary into
coherent multi-step storylines.  FinalCommentary carries the final
text alongside suppression decisions, progression metadata, and a
compact race-state summary for downstream consumers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.candidate_event import EventType


# ── Progression stages ───────────────────────────────────────────────────

class ProgressionStage(str, Enum):
    """Narrative progression stage within a storyline."""

    INITIAL_INTEREST = "initial_interest"
    ACTIVE_PRESSURE = "active_pressure"
    IMMINENT_ACTION = "imminent_action"
    DECISIVE_EVENT = "decisive_event"
    CONCLUDED = "concluded"


# ── Suppression ──────────────────────────────────────────────────────────

class FinalSuppressionReason(str, Enum):
    """Why a generated commentary was suppressed at the final stage."""

    NONE = "none"
    DUPLICATE_SEMANTICS = "duplicate_semantics"
    NO_PROGRESSION = "no_progression"
    REFRESH_INTERVAL = "refresh_interval"
    GENERATION_FAILURE = "generation_failure"


# ── Confidence band ──────────────────────────────────────────────────────

class ConfidenceBand(str, Enum):
    """Coarse confidence in the final commentary's quality."""

    HIGH = "high"        # strong alignment + clear progression
    MEDIUM = "medium"    # reasonable but could be better
    LOW = "low"          # fallback or uncertain state
    SUPPRESSED = "suppressed"  # commentary was suppressed


# ── Compact race-state summary ──────────────────────────────────────────

class RaceStateSummary(BaseModel):
    """Minimal race-state snapshot embedded in final output."""

    lap: Optional[int] = None
    leader: Optional[str] = None
    flag_state: Optional[str] = None
    safety_car: bool = False
    involved_driver_summary: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-driver: {position, gap_ahead_sec, tire_compound, tire_age_laps}",
    )
    battle_priority: Optional[float] = Field(
        default=None,
        description="Highest battle intensity among involved drivers",
    )


# ── Final commentary ────────────────────────────────────────────────────

class FinalCommentary(BaseModel):
    """One final commentary output — the pipeline's end product."""

    final_id: str
    timestamp: datetime
    beat_id: str
    storyline_id: Optional[str] = None
    event_type: EventType
    involved_drivers: list[str] = Field(default_factory=list)

    # Text.
    final_text: str = Field(
        description="Post-processed, progression-aware commentary text",
    )

    # Quality and context.
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    replay_lag_seconds: float = Field(
        default=0.0,
        description="Lag between event timestamp and commentary emission",
    )
    race_state_summary: RaceStateSummary = Field(default_factory=RaceStateSummary)

    # Suppression.
    suppression_applied: bool = False
    suppression_reason: FinalSuppressionReason = FinalSuppressionReason.NONE

    # Progression.
    progression_stage: ProgressionStage = ProgressionStage.INITIAL_INTEREST

    # Provenance.
    source_generation_id: Optional[str] = Field(
        default=None,
        description="Links back to GeneratedCommentary.generation_id",
    )

    model_config = {"ser_json_timedelta": "float"}
