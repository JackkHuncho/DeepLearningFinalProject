"""In-memory RaceStateEngine — maintains hot per-driver and per-battle state.

Consumes TelemetryFrame objects one at a time and maintains running state
that the FeatureExtractor queries to build RaceStateSnapshots.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Optional

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    TelemetryFrame,
)

logger = logging.getLogger(__name__)

# How many recent lap times to keep per driver for rolling statistics.
PACE_WINDOW = 5

# Gap threshold (seconds) to consider two cars "in a battle".
BATTLE_GAP_THRESHOLD = 2.0

# Pit-window heuristic thresholds (laps) per compound.
PIT_WINDOW_THRESHOLDS: dict[str, int] = {
    "SOFT": 15,
    "MEDIUM": 22,
    "HARD": 30,
    "INTERMEDIATE": 20,
    "WET": 25,
}


class _DriverHistory:
    """Mutable per-driver accumulator kept by the engine."""

    __slots__ = (
        "driver_number",
        "abbreviation",
        "position",
        "lap_number",
        "gap_ahead_sec",
        "tire_compound",
        "tire_age_laps",
        "stint_length",
        "drs_active",
        "speed_kph",
        "lap_times",
        "prev_compound",
    )

    def __init__(self, driver_number: str, abbreviation: str) -> None:
        self.driver_number = driver_number
        self.abbreviation = abbreviation
        self.position: Optional[int] = None
        self.lap_number: Optional[int] = None
        self.gap_ahead_sec: Optional[float] = None
        self.tire_compound: Optional[str] = None
        self.tire_age_laps: Optional[int] = None
        self.stint_length: int = 0
        self.drs_active: Optional[bool] = None
        self.speed_kph: Optional[float] = None
        # Rolling window of recent lap times (seconds).  Derived from
        # speed_kph as a proxy if actual lap time is unavailable; when
        # coming from sample data that only has speed, we store the
        # inter-frame gap-ahead delta as a stand-in.
        self.lap_times: deque[float] = deque(maxlen=PACE_WINDOW)
        self.prev_compound: Optional[str] = None

    def update(self, ds: DriverState) -> None:
        """Ingest one TelemetryFrame's DriverState."""
        self.position = ds.position
        self.gap_ahead_sec = ds.gap_ahead_sec
        self.drs_active = ds.drs_active
        self.speed_kph = ds.speed_kph

        # Detect compound change → reset stint length.
        if ds.tire_compound is not None and ds.tire_compound != self.prev_compound:
            if self.prev_compound is not None:
                # Compound actually changed → pit stop happened.
                self.stint_length = 0
            self.prev_compound = ds.tire_compound
        self.tire_compound = ds.tire_compound

        if ds.tire_age_laps is not None:
            self.tire_age_laps = ds.tire_age_laps
            # Use tire age as stint proxy when available.
            self.stint_length = ds.tire_age_laps

        # Lap progression and pace tracking.
        if ds.lap_number is not None:
            old_lap = self.lap_number
            self.lap_number = ds.lap_number
            # If we have speed, store an approximate "lap-time" proxy.
            # Formula:  proxy ≈ (track_length_km / speed_kph) * 3600
            # We use a fixed 5.793 km (Monza) as a rough constant; the
            # *absolute* value doesn't matter for trend / delta analysis,
            # only relative changes between laps.
            if ds.speed_kph is not None and ds.speed_kph > 0:
                proxy_sec = (5.793 / ds.speed_kph) * 3600.0
                if old_lap is not None and ds.lap_number != old_lap:
                    self.lap_times.append(proxy_sec)


class _BattleTracker:
    """Tracks how long pairs of drivers have been within battle range."""

    def __init__(self) -> None:
        # (driver_ahead_abbr, driver_behind_abbr) → consecutive frame count
        self._durations: dict[tuple[str, str], int] = defaultdict(int)

    def update(self, active_pairs: set[tuple[str, str]]) -> None:
        """Increment active pairs, reset inactive ones."""
        to_delete = [k for k in self._durations if k not in active_pairs]
        for k in to_delete:
            del self._durations[k]
        for pair in active_pairs:
            self._durations[pair] += 1

    def duration(self, ahead: str, behind: str) -> int:
        return self._durations.get((ahead, behind), 0)


class RaceStateEngine:
    """Stateful engine that accumulates TelemetryFrame data.

    Call ``update(frame)`` for each incoming frame.  Then query the engine
    for current driver histories, battle durations, and global state to
    feed into the FeatureExtractor.
    """

    def __init__(self) -> None:
        self.drivers: dict[str, _DriverHistory] = {}
        self.battles = _BattleTracker()

        # Global state (latest values).
        self.track_status: Optional[str] = None
        self.safety_car: Optional[bool] = None
        self.air_temp: Optional[float] = None
        self.track_temp: Optional[float] = None
        self.humidity: Optional[float] = None
        self.rainfall: Optional[bool] = None
        self.wind_speed: Optional[float] = None
        self.session_clock: Optional[str] = None

        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: TelemetryFrame) -> None:
        """Ingest one TelemetryFrame, updating all internal state."""
        self._frame_count += 1

        # Update per-driver state.
        for ds in frame.drivers:
            key = ds.abbreviation
            if key not in self.drivers:
                self.drivers[key] = _DriverHistory(ds.driver_number, ds.abbreviation)
            self.drivers[key].update(ds)

        # Update global state.
        gs = frame.global_state
        if gs.track_status is not None:
            self.track_status = gs.track_status
        if gs.safety_car is not None:
            self.safety_car = gs.safety_car
        if gs.air_temp_celsius is not None:
            self.air_temp = gs.air_temp_celsius
        if gs.track_temp_celsius is not None:
            self.track_temp = gs.track_temp_celsius
        if gs.humidity_pct is not None:
            self.humidity = gs.humidity_pct
        if gs.rainfall is not None:
            self.rainfall = gs.rainfall
        if gs.wind_speed_ms is not None:
            self.wind_speed = gs.wind_speed_ms
        if gs.session_clock is not None:
            self.session_clock = gs.session_clock

        # Detect active battles.
        self._update_battles()

    def get_driver(self, abbr: str) -> Optional[_DriverHistory]:
        return self.drivers.get(abbr)

    def sorted_drivers(self) -> list[_DriverHistory]:
        """Return drivers sorted by position (None-position at end)."""
        return sorted(
            self.drivers.values(),
            key=lambda d: d.position if d.position is not None else 999,
        )

    # ------------------------------------------------------------------
    # Battle detection
    # ------------------------------------------------------------------

    def _update_battles(self) -> None:
        """Identify pairs within BATTLE_GAP_THRESHOLD and update tracker."""
        ordered = self.sorted_drivers()
        active: set[tuple[str, str]] = set()
        for i in range(1, len(ordered)):
            behind = ordered[i]
            ahead = ordered[i - 1]
            gap = behind.gap_ahead_sec
            if gap is not None and 0 < gap <= BATTLE_GAP_THRESHOLD:
                active.add((ahead.abbreviation, behind.abbreviation))
        self.battles.update(active)
