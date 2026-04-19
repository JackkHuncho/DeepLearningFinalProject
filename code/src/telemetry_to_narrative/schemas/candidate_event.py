"""Temporary CandidateEvent schema for Phase 5 testing.

This stands in for the Phase 4 event-recognition output.  When Phase 4 is
ready, the scheduler should accept the real CandidateEvent objects; the
fields below are a compatible subset.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Coarse event categories used for editorial precedence."""

    RACE_CONTROL = "race_control"          # safety car, red flag, VSC
    LEAD_BATTLE = "lead_battle"            # battle for P1
    PODIUM_BATTLE = "podium_battle"        # battle for P2-P3
    INCIDENT = "incident"                  # collision, penalty, DNF
    PIT_STRATEGY = "pit_strategy"          # pit stop affecting top positions
    OVERTAKE = "overtake"                  # position change
    MIDFIELD_BATTLE = "midfield_battle"    # battle outside top 3
    STRATEGIC_DEVELOPMENT = "strategic"    # weather change, tyre deg shift
    TELEMETRY_CHANGE = "telemetry_change"  # low-impact speed / gap change


class CandidateEvent(BaseModel):
    """A recognized event candidate that the scheduler will prioritize.

    This temporary schema mirrors what Phase 4 will produce.  The
    EditorialScheduler is generic over any object that exposes these
    fields; when Phase 4 lands, swap the import.
    """

    event_id: str = Field(description="Unique identifier for this event")
    timestamp: datetime
    event_type: EventType
    involved_drivers: list[str] = Field(
        default_factory=list,
        description="Driver abbreviations involved",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Recognition confidence from the detector (1.0 = certain)",
    )
    source_frame_id: int = Field(
        default=0, ge=0,
        description="Frame ID of the snapshot that produced this event",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the event",
    )

    # Contextual hints the scheduler uses for scoring.
    position_of_interest: Optional[int] = Field(
        default=None,
        description="Highest position involved (lower = more important)",
    )
    gap_sec: Optional[float] = Field(
        default=None,
        description="Gap involved in the event (smaller = more urgent)",
    )
    battle_intensity: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Battle intensity score from FeatureExtractor",
    )
    safety_critical: bool = Field(
        default=False,
        description="True for race-control / safety events",
    )
