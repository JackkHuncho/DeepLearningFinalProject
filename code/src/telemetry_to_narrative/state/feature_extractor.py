"""FeatureExtractor — consumes TelemetryFrame, returns RaceStateSnapshot.

Relies on a RaceStateEngine for accumulated state and computes all derived
features defined in the RaceStateSnapshot schema.

Feature formulas and heuristics
===============================

recent_pace_sec
    Rolling mean of the last PACE_WINDOW lap-time proxies stored per driver.

pace_trend
    Ordinary-least-squares slope of lap-time proxies over PACE_WINDOW.
    Positive slope → driver is slowing down (degradation or traffic).
    Negative slope → driver is speeding up (clear air, fresh tyres).

    Formula:  slope = Σ((i - ī)(t_i - t̄)) / Σ((i - ī)²)
    where i is the lap index within the window.

relative_pace_delta
    (this_driver.recent_pace - car_ahead.recent_pace).
    Negative → this driver is lapping faster than the car ahead → closing.

catch_time_laps
    gap_ahead_sec / abs(relative_pace_delta)  when delta < 0 (catching).
    Represents laps-to-overtake at current closing rate.
    Clamped to [0, 999] to avoid infinities.

degradation_score
    (latest_lap_time - rolling_mean) / rolling_mean.
    Positive → current pace is worse than the rolling average.
    Typical range [-0.05, 0.10].

pit_window_open
    Heuristic: tire_age_laps >= compound-specific threshold.
    SOFT ≥ 15, MEDIUM ≥ 22, HARD ≥ 30, INTER ≥ 20, WET ≥ 25.
    Missing compound or age → None.

battle_intensity
    Composite score in [0, 1]:
        proximity_factor = max(0, 1 - gap / BATTLE_GAP_THRESHOLD)
        pace_factor      = clamp(abs(pace_delta) * 5, 0, 1)  if catching
        drs_bonus        = 0.15 if DRS active else 0
        intensity        = 0.50 * proximity_factor
                         + 0.35 * pace_factor
                         + 0.15 * drs_bonus_norm
    where drs_bonus_norm = 1.0 when DRS active, 0 otherwise.
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.schemas.race_state import (
    BattleState,
    DerivedSummary,
    DriverRaceState,
    GlobalRaceStateSummary,
    RaceStateSnapshot,
)
from src.telemetry_to_narrative.state.race_state_engine import (
    BATTLE_GAP_THRESHOLD,
    PIT_WINDOW_THRESHOLDS,
    RaceStateEngine,
    _DriverHistory,
)

logger = logging.getLogger(__name__)


def _ols_slope(values: list[float]) -> Optional[float]:
    """Compute OLS slope of values over indices [0..n-1].

    Returns None if fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class FeatureExtractor:
    """Stateless transformer: (engine state + frame) → RaceStateSnapshot.

    Usage::

        engine = RaceStateEngine()
        extractor = FeatureExtractor(engine)

        for frame in replay:
            engine.update(frame)
            snapshot = extractor.extract(frame)
    """

    def __init__(self, engine: RaceStateEngine) -> None:
        self.engine = engine

    def extract(self, frame: TelemetryFrame) -> RaceStateSnapshot:
        """Build a full RaceStateSnapshot from the current engine state."""
        drivers_map: dict[str, DriverRaceState] = {}
        ordered = self.engine.sorted_drivers()

        # Pre-compute a lookup of recent pace by abbreviation so we can
        # compute relative pace deltas between adjacent drivers.
        pace_lookup: dict[str, Optional[float]] = {}
        for dh in ordered:
            pace_lookup[dh.abbreviation] = self._recent_pace(dh)

        # Build per-driver state.
        for idx, dh in enumerate(ordered):
            # Determine gap_behind from the *next* driver in position order.
            gap_behind: Optional[float] = None
            if idx < len(ordered) - 1:
                next_driver = ordered[idx + 1]
                gap_behind = next_driver.gap_ahead_sec

            # Determine the car ahead's abbreviation for relative pace.
            ahead_abbr: Optional[str] = None
            if idx > 0:
                ahead_abbr = ordered[idx - 1].abbreviation

            recent_pace = pace_lookup.get(dh.abbreviation)
            pace_trend = self._pace_trend(dh)
            relative_delta = self._relative_pace_delta(
                recent_pace, pace_lookup.get(ahead_abbr) if ahead_abbr else None
            )
            catch_time = self._catch_time(dh.gap_ahead_sec, relative_delta)
            degradation = self._degradation_score(dh)
            pit_open = self._pit_window_open(dh.tire_compound, dh.tire_age_laps)

            drs = DriverRaceState(
                driver_number=dh.driver_number,
                abbreviation=dh.abbreviation,
                position=dh.position,
                lap_number=dh.lap_number,
                gap_ahead_sec=dh.gap_ahead_sec,
                gap_behind_sec=gap_behind,
                tire_compound=dh.tire_compound,
                tire_age_laps=dh.tire_age_laps,
                stint_length=dh.stint_length,
                drs_active=dh.drs_active,
                speed_kph=dh.speed_kph,
                recent_pace_sec=round(recent_pace, 3) if recent_pace is not None else None,
                pace_trend=round(pace_trend, 4) if pace_trend is not None else None,
                relative_pace_delta=(
                    round(relative_delta, 4) if relative_delta is not None else None
                ),
                catch_time_laps=(
                    round(catch_time, 2) if catch_time is not None else None
                ),
                degradation_score=(
                    round(degradation, 5) if degradation is not None else None
                ),
                pit_window_open=pit_open,
            )
            drivers_map[dh.abbreviation] = drs

        # Build battle summaries.
        battles = self._build_battles(ordered, pace_lookup, drivers_map)

        # Build global state.
        global_state = GlobalRaceStateSummary(
            track_status=self.engine.track_status,
            safety_car=self.engine.safety_car,
            virtual_safety_car=(
                self.engine.track_status is not None
                and "vsc" in (self.engine.track_status or "").lower()
            ) or None,
            red_flag=(
                self.engine.track_status is not None
                and "red" in (self.engine.track_status or "").lower()
            ) or None,
            air_temp_celsius=self.engine.air_temp,
            track_temp_celsius=self.engine.track_temp,
            humidity_pct=self.engine.humidity,
            rainfall=self.engine.rainfall,
            wind_speed_ms=self.engine.wind_speed,
            session_clock=self.engine.session_clock,
            race_control_message=self.engine.track_status,
        )

        # Build derived summary.
        paces = [p for p in pace_lookup.values() if p is not None]
        fastest_abbr: Optional[str] = None
        fastest_pace: Optional[float] = None
        if paces:
            fastest_pace = min(paces)
            fastest_abbr = next(
                (k for k, v in pace_lookup.items() if v == fastest_pace), None
            )

        leader_abbr: Optional[str] = None
        if ordered:
            leader_abbr = ordered[0].abbreviation

        summary = DerivedSummary(
            leader=leader_abbr,
            total_drivers=len(ordered),
            active_battles=len(battles),
            safety_car_active=bool(self.engine.safety_car),
            avg_field_pace_sec=(
                round(statistics.mean(paces), 3) if paces else None
            ),
            fastest_driver=fastest_abbr,
            fastest_pace_sec=(
                round(fastest_pace, 3) if fastest_pace is not None else None
            ),
        )

        return RaceStateSnapshot(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            session_time=frame.session_time,
            session_info=frame.session_info,
            global_state=global_state,
            drivers=drivers_map,
            active_battles=battles,
            derived_summary=summary,
        )

    # ------------------------------------------------------------------
    # Derived-feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recent_pace(dh: _DriverHistory) -> Optional[float]:
        """Rolling mean lap-time proxy over the window."""
        if not dh.lap_times:
            return None
        return statistics.mean(dh.lap_times)

    @staticmethod
    def _pace_trend(dh: _DriverHistory) -> Optional[float]:
        """OLS slope of lap-time proxy over window (sec / lap)."""
        if len(dh.lap_times) < 2:
            return None
        return _ols_slope(list(dh.lap_times))

    @staticmethod
    def _relative_pace_delta(
        my_pace: Optional[float], ahead_pace: Optional[float]
    ) -> Optional[float]:
        """(my_pace - ahead_pace).  Negative means I am faster."""
        if my_pace is None or ahead_pace is None:
            return None
        return my_pace - ahead_pace

    @staticmethod
    def _catch_time(
        gap_sec: Optional[float], pace_delta: Optional[float]
    ) -> Optional[float]:
        """Laps to catch the car ahead.

        Only meaningful when pace_delta < 0 (I am faster) and gap > 0.
        """
        if gap_sec is None or pace_delta is None:
            return None
        if pace_delta >= 0:
            # Not catching → undefined
            return None
        if gap_sec <= 0:
            return 0.0
        if abs(pace_delta) < 1e-6:
            return 999.0
        raw = gap_sec / abs(pace_delta)
        return _clamp(raw, 0.0, 999.0)

    @staticmethod
    def _degradation_score(dh: _DriverHistory) -> Optional[float]:
        """(latest - rolling_mean) / rolling_mean."""
        if len(dh.lap_times) < 2:
            return None
        mean_val = statistics.mean(dh.lap_times)
        if mean_val == 0:
            return None
        latest = dh.lap_times[-1]
        return (latest - mean_val) / mean_val

    @staticmethod
    def _pit_window_open(
        compound: Optional[str], tire_age: Optional[int]
    ) -> Optional[bool]:
        """Heuristic: is tire age past the compound-specific threshold?"""
        if compound is None or tire_age is None:
            return None
        threshold = PIT_WINDOW_THRESHOLDS.get(compound.upper())
        if threshold is None:
            return None
        return tire_age >= threshold

    # ------------------------------------------------------------------
    # Battle construction
    # ------------------------------------------------------------------

    def _build_battles(
        self,
        ordered: list[_DriverHistory],
        pace_lookup: dict[str, Optional[float]],
        drivers_map: dict[str, DriverRaceState],
    ) -> list[BattleState]:
        """Identify and score active battles."""
        battles: list[BattleState] = []

        for i in range(1, len(ordered)):
            behind = ordered[i]
            ahead = ordered[i - 1]
            gap = behind.gap_ahead_sec
            if gap is None or gap <= 0 or gap > BATTLE_GAP_THRESHOLD:
                continue

            pace_behind = pace_lookup.get(behind.abbreviation)
            pace_ahead = pace_lookup.get(ahead.abbreviation)
            pace_delta: Optional[float] = None
            if pace_behind is not None and pace_ahead is not None:
                pace_delta = pace_behind - pace_ahead

            drs = behind.drs_active
            duration = self.engine.battles.duration(
                ahead.abbreviation, behind.abbreviation
            )

            # Battle intensity composite score.
            # proximity_factor: 1.0 when gap=0, 0.0 when gap=threshold
            proximity = max(0.0, 1.0 - gap / BATTLE_GAP_THRESHOLD)

            # pace_factor: how fast the chaser is closing.  Scale so that
            # a 0.2s/lap closing rate maps to ~1.0.
            pace_factor = 0.0
            if pace_delta is not None and pace_delta < 0:
                pace_factor = _clamp(abs(pace_delta) * 5.0, 0.0, 1.0)

            drs_norm = 1.0 if drs else 0.0

            intensity = (
                0.50 * proximity + 0.35 * pace_factor + 0.15 * drs_norm
            )

            battles.append(
                BattleState(
                    driver_ahead=ahead.abbreviation,
                    driver_behind=behind.abbreviation,
                    gap_sec=round(gap, 3),
                    pace_delta_sec=(
                        round(pace_delta, 4) if pace_delta is not None else None
                    ),
                    drs_available=drs,
                    battle_intensity=round(intensity, 4),
                    battle_duration_laps=duration,
                )
            )

        return battles
