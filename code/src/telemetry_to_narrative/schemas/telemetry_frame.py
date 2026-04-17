"""Stable TelemetryFrame schema for timestamped race snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field


class DriverState(BaseModel):
    """Per-driver telemetry at a single point in time."""

    driver_number: str
    abbreviation: str
    position: Optional[int] = None
    lap_number: Optional[int] = None
    gap_ahead_sec: Optional[float] = Field(
        default=None, description="Gap to car ahead in seconds"
    )
    tire_compound: Optional[str] = None
    tire_age_laps: Optional[int] = None
    drs_active: Optional[bool] = None
    speed_kph: Optional[float] = None


class GlobalState(BaseModel):
    """Session-wide state at a single point in time."""

    track_status: Optional[str] = Field(
        default=None,
        description="Flag / race control status (e.g. Green, Yellow, SC, Red)",
    )
    safety_car: Optional[bool] = None
    air_temp_celsius: Optional[float] = None
    track_temp_celsius: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall: Optional[bool] = None
    wind_speed_ms: Optional[float] = None
    session_clock: Optional[str] = Field(
        default=None, description="Elapsed session time as HH:MM:SS"
    )


class SessionInfo(BaseModel):
    """Metadata identifying the session being replayed."""

    year: int
    grand_prix: str
    session_type: str  # FP1, FP2, FP3, Q, R


class TelemetryFrame(BaseModel):
    """One timestamped snapshot of the race / session state."""

    frame_id: int = Field(ge=0, description="Monotonically increasing frame counter")
    timestamp: datetime = Field(description="UTC wall-clock time of this snapshot")
    session_time: Optional[timedelta] = Field(
        default=None, description="Time elapsed since session start"
    )
    session_info: SessionInfo
    drivers: list[DriverState] = Field(default_factory=list)
    global_state: GlobalState = Field(default_factory=GlobalState)

    model_config = {"ser_json_timedelta": "float"}
