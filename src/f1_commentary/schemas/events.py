"""Typed event schemas for the F1 commentary pipeline.

Defines Pydantic v2 models for every event type that flows through the
event bus (telemetry, race state, overtakes, pit-stops, flags, etc.).
"""

from datetime import datetime
from enum import Enum
from typing import Optional

import uuid

from pydantic import BaseModel, Field


class EventType(str, Enum):
    TELEMETRY_FRAME = "telemetry_frame"
    RACE_STATE_UPDATE = "race_state_update"
    CANDIDATE_EVENT = "candidate_event"
    SCHEDULED_BEAT = "scheduled_beat"
    RETRIEVAL_RESULT = "retrieval_result"
    GENERATED_COMMENTARY = "generated_commentary"
    GUARDED_COMMENTARY = "guarded_commentary"
    FINAL_COMMENTARY = "final_commentary"


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    event_type: EventType


class DriverState(BaseModel):
    driver_id: str  # 3-letter code e.g. "HAM"
    position: int
    lap_number: int
    gap_ahead: Optional[float] = None  # seconds
    tire_compound: Optional[str] = None  # SOFT/MEDIUM/HARD/INTER/WET
    tire_age: Optional[int] = None  # laps on current set
    drs_available: Optional[bool] = None
    speed: Optional[float] = None  # km/h
    last_lap_time: Optional[float] = None  # seconds


class TelemetryFrame(BaseEvent):
    event_type: EventType = EventType.TELEMETRY_FRAME
    session_key: str  # e.g. "2024_Monza_R"
    lap: int
    drivers: list[DriverState]
    flag_status: Optional[str] = None  # GREEN/YELLOW/RED/SC/VSC
    weather: Optional[str] = None
    session_clock: Optional[float] = None  # seconds elapsed


class RaceStateSnapshot(BaseEvent):
    """Enriched state with derived features. Produced by feature extraction (Phase 3)."""

    event_type: EventType = EventType.RACE_STATE_UPDATE
    session_key: str
    lap: int
    drivers: list[DriverState]
    # Derived features
    gaps: dict[str, float] = Field(default_factory=dict)
    pace_deltas: dict[str, float] = Field(default_factory=dict)
    catch_times: dict[str, Optional[float]] = Field(default_factory=dict)
    stint_deg_scores: dict[str, float] = Field(default_factory=dict)
    battle_pairs: list[tuple[str, str]] = Field(default_factory=list)
    flag_status: Optional[str] = None
    weather: Optional[str] = None


class CandidateEventType(str, Enum):
    OVERTAKE_OPPORTUNITY = "overtake_opportunity"
    DRS_THREAT = "drs_threat"
    ACTIVE_BATTLE = "active_battle"
    PIT_STOP = "pit_stop"
    PIT_STRATEGY_TRIGGER = "pit_strategy_trigger"
    TIRE_DEGRADATION = "tire_degradation"
    SAFETY_CAR = "safety_car"
    YELLOW_FLAG = "yellow_flag"
    RED_FLAG = "red_flag"
    LEAD_BATTLE = "lead_battle"
    MIDFIELD_BATTLE = "midfield_battle"


class CandidateEvent(BaseEvent):
    event_type: EventType = EventType.CANDIDATE_EVENT
    candidate_type: CandidateEventType
    involved_drivers: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0, default=0.5)
    explanation: str = ""
    triggering_features: dict = Field(default_factory=dict)
    storyline_id: Optional[str] = None
    lap: int = 0
    session_key: str = ""


class ScheduledBeat(BaseEvent):
    event_type: EventType = EventType.SCHEDULED_BEAT
    source_event_id: str
    candidate_type: CandidateEventType
    priority_score: float
    involved_drivers: list[str]
    race_phase: str = ""  # "early", "mid", "late"
    storyline_id: Optional[str] = None
    deferred: bool = False
    suppress_reason: Optional[str] = None


class RetrievalResult(BaseEvent):
    event_type: EventType = EventType.RETRIEVAL_RESULT
    query_event_id: str
    tier: int  # 1 or 2
    chunks: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0
    gated_out: bool = False


class GeneratedCommentary(BaseEvent):
    event_type: EventType = EventType.GENERATED_COMMENTARY
    source_beat_id: str
    prompt_text: str
    generated_text: str
    model_variant: str = ""
    generation_latency_ms: float = 0.0
    involved_drivers: list[str] = Field(default_factory=list)


class ClaimSupport(BaseModel):
    claim: str
    support_type: str  # "observed", "inferred", "retrieved", "unsupported"
    confidence: float = 0.5
    evidence: Optional[str] = None


class GuardedCommentary(BaseEvent):
    event_type: EventType = EventType.GUARDED_COMMENTARY
    source_commentary_id: str
    original_text: str
    adjusted_text: str
    claims: list[ClaimSupport] = Field(default_factory=list)
    confidence_band: str = "medium"  # "high", "medium", "low"


class FinalCommentary(BaseEvent):
    event_type: EventType = EventType.FINAL_COMMENTARY
    storyline_id: Optional[str] = None
    final_text: str
    confidence_band: str = "medium"
    replay_lag_ms: float = 0.0
    race_state_summary: dict = Field(default_factory=dict)
    involved_drivers: list[str] = Field(default_factory=list)


# Mapping from EventType to the concrete model class, used for deserialization.
EVENT_TYPE_MODEL_MAP: dict[EventType, type[BaseEvent]] = {
    EventType.TELEMETRY_FRAME: TelemetryFrame,
    EventType.RACE_STATE_UPDATE: RaceStateSnapshot,
    EventType.CANDIDATE_EVENT: CandidateEvent,
    EventType.SCHEDULED_BEAT: ScheduledBeat,
    EventType.RETRIEVAL_RESULT: RetrievalResult,
    EventType.GENERATED_COMMENTARY: GeneratedCommentary,
    EventType.GUARDED_COMMENTARY: GuardedCommentary,
    EventType.FINAL_COMMENTARY: FinalCommentary,
}
