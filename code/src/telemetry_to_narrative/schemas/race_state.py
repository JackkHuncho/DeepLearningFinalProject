"""RaceStateSnapshot schema — model-facing representation of race state.

This is the output of the FeatureExtractor: a single, self-contained snapshot
that downstream consumers (commentary generation, retrieval, evaluation) can
work with without needing raw telemetry.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo


class DriverRaceState(BaseModel):
    """Derived per-driver state at a single point in time.

    Contains both raw forwarded fields (position, gap, tire) and computed
    features (pace trend, catch-time, degradation).
    """

    driver_number: str
    abbreviation: str

    # --- forwarded from TelemetryFrame ---
    position: Optional[int] = None
    lap_number: Optional[int] = None
    gap_ahead_sec: Optional[float] = Field(
        default=None, description="Gap to the car immediately ahead (seconds)"
    )
    gap_behind_sec: Optional[float] = Field(
        default=None, description="Gap to the car immediately behind (seconds)"
    )
    tire_compound: Optional[str] = None
    tire_age_laps: Optional[int] = None
    stint_length: Optional[int] = Field(
        default=None,
        description="Laps since last compound change (same as tire_age if no pit)",
    )
    drs_active: Optional[bool] = None
    speed_kph: Optional[float] = None

    # --- derived features ---
    recent_pace_sec: Optional[float] = Field(
        default=None,
        description="Rolling mean lap time over the last N laps (seconds)",
    )
    pace_trend: Optional[float] = Field(
        default=None,
        description=(
            "Linear slope of lap times over last N laps (seconds / lap). "
            "Positive = slowing down, negative = speeding up."
        ),
    )
    relative_pace_delta: Optional[float] = Field(
        default=None,
        description=(
            "Difference between this driver's recent pace and the car ahead's "
            "recent pace (seconds). Negative = this driver is faster."
        ),
    )
    catch_time_laps: Optional[float] = Field(
        default=None,
        description=(
            "Estimated laps to catch the car ahead, computed as "
            "gap_ahead / abs(relative_pace_delta) when delta is negative (catching)."
        ),
    )
    degradation_score: Optional[float] = Field(
        default=None,
        description=(
            "Pace drop relative to rolling baseline. Computed as "
            "(latest_lap_time - rolling_mean) / rolling_mean. "
            "Positive = degrading. Range roughly [-0.05, 0.10]."
        ),
    )
    pit_window_open: Optional[bool] = Field(
        default=None,
        description=(
            "Heuristic flag: True if tire age >= compound threshold. "
            "SOFT >= 15 laps, MEDIUM >= 22 laps, HARD >= 30 laps."
        ),
    )


class BattleState(BaseModel):
    """Summary of an active on-track battle between two drivers."""

    driver_ahead: str = Field(description="Abbreviation of the leading driver")
    driver_behind: str = Field(description="Abbreviation of the chasing driver")
    gap_sec: Optional[float] = Field(
        default=None, description="Time gap between the two cars"
    )
    pace_delta_sec: Optional[float] = Field(
        default=None,
        description=(
            "Pace difference (behind - ahead). "
            "Negative means the chasing driver is faster."
        ),
    )
    drs_available: Optional[bool] = Field(
        default=None,
        description="Whether the chasing driver has DRS",
    )
    battle_intensity: Optional[float] = Field(
        default=None,
        description=(
            "Composite score [0, 1] combining proximity, pace convergence, "
            "and DRS. Higher = more intense / likely overtake."
        ),
    )
    battle_duration_laps: Optional[int] = Field(
        default=None,
        description="How many consecutive frames these two have been within battle threshold",
    )


class GlobalRaceStateSummary(BaseModel):
    """High-level session state derived from telemetry global state."""

    track_status: Optional[str] = None
    safety_car: Optional[bool] = None
    virtual_safety_car: Optional[bool] = None
    red_flag: Optional[bool] = None
    air_temp_celsius: Optional[float] = None
    track_temp_celsius: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall: Optional[bool] = None
    wind_speed_ms: Optional[float] = None
    session_clock: Optional[str] = None
    race_control_message: Optional[str] = Field(
        default=None, description="Most recent race control message text"
    )


class DerivedSummary(BaseModel):
    """Compact session-level derived fields for quick inspection."""

    leader: Optional[str] = None
    total_drivers: int = 0
    active_battles: int = 0
    safety_car_active: bool = False
    avg_field_pace_sec: Optional[float] = Field(
        default=None, description="Mean recent pace across all drivers"
    )
    fastest_driver: Optional[str] = None
    fastest_pace_sec: Optional[float] = None


class RaceStateSnapshot(BaseModel):
    """Complete race state at a single point in time.

    This is the primary output of the FeatureExtractor / RaceStateEngine
    and the primary input for downstream consumers.
    """

    frame_id: int = Field(ge=0)
    timestamp: datetime
    session_time: Optional[timedelta] = None
    session_info: SessionInfo
    global_state: GlobalRaceStateSummary = Field(
        default_factory=GlobalRaceStateSummary
    )
    drivers: dict[str, DriverRaceState] = Field(
        default_factory=dict,
        description="Keyed by driver abbreviation",
    )
    active_battles: list[BattleState] = Field(default_factory=list)
    derived_summary: DerivedSummary = Field(default_factory=DerivedSummary)

    model_config = {"ser_json_timedelta": "float"}
