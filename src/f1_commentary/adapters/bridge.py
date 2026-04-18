"""Bridge between even-phase (``f1_commentary``) and odd-phase
(``telemetry_to_narrative``) schema systems.

Usage
-----
>>> from f1_commentary.adapters.bridge import odd_telemetry_to_even

All adapter functions accept plain model instances from one side and return
validated model instances for the other side.

PYTHONPATH note
~~~~~~~~~~~~~~~
The odd-phase package resolves its own internal imports via
``src.telemetry_to_narrative.…``, which requires ``code/`` on
``sys.path``.  This module inserts that path automatically.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Ensure ``code/`` is importable so that the odd-phase schemas work.
# The odd-phase package uses ``from src.telemetry_to_narrative...`` where
# ``src`` is the ``code/src`` directory (which has __init__.py).
# ---------------------------------------------------------------------------
_CODE_ROOT = str(Path(__file__).resolve().parents[3] / "code")
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

# ---------------------------------------------------------------------------
# Even-phase imports  (always available)
# ---------------------------------------------------------------------------
from f1_commentary.schemas.events import (
    BaseEvent,
    CandidateEvent as EvenCandidateEvent,
    CandidateEventType,
    DriverState as EvenDriverState,
    EventType as EvenEventType,
    FinalCommentary as EvenFinalCommentary,
    GeneratedCommentary as EvenGeneratedCommentary,
    GuardedCommentary as EvenGuardedCommentary,
    RaceStateSnapshot as EvenRaceStateSnapshot,
    TelemetryFrame as EvenTelemetryFrame,
)

# ---------------------------------------------------------------------------
# Odd-phase imports  (need code/src on sys.path)
# ---------------------------------------------------------------------------
from src.telemetry_to_narrative.schemas.telemetry_frame import (
    TelemetryFrame as OddTelemetryFrame,
    DriverState as OddTelemetryDriverState,
    GlobalState as OddGlobalState,
    SessionInfo as OddSessionInfo,
)
from src.telemetry_to_narrative.schemas.race_state import (
    RaceStateSnapshot as OddRaceStateSnapshot,
    DriverRaceState as OddDriverRaceState,
    BattleState as OddBattleState,
    GlobalRaceStateSummary as OddGlobalRaceStateSummary,
)
from src.telemetry_to_narrative.schemas.candidate_event import (
    CandidateEvent as OddCandidateEvent,
    EventType as OddEventType,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScheduledBeat as OddScheduledBeat,
)
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary as OddGeneratedCommentary,
)
from src.telemetry_to_narrative.schemas.final_commentary import (
    FinalCommentary as OddFinalCommentary,
    ConfidenceBand as OddConfidenceBand,
    RaceStateSummary as OddRaceStateSummary,
)
from src.telemetry_to_narrative.schemas.baseline_commentary import (
    BaselineCommentary as OddBaselineCommentary,
)

# =========================================================================
# Event-type mapping tables
# =========================================================================

EVEN_TO_ODD_EVENT_TYPE: Dict[CandidateEventType, OddEventType] = {
    CandidateEventType.OVERTAKE_OPPORTUNITY: OddEventType.OVERTAKE,
    CandidateEventType.DRS_THREAT: OddEventType.OVERTAKE,
    CandidateEventType.ACTIVE_BATTLE: OddEventType.MIDFIELD_BATTLE,
    CandidateEventType.LEAD_BATTLE: OddEventType.LEAD_BATTLE,
    CandidateEventType.MIDFIELD_BATTLE: OddEventType.MIDFIELD_BATTLE,
    CandidateEventType.PIT_STOP: OddEventType.PIT_STRATEGY,
    CandidateEventType.PIT_STRATEGY_TRIGGER: OddEventType.PIT_STRATEGY,
    CandidateEventType.TIRE_DEGRADATION: OddEventType.STRATEGIC_DEVELOPMENT,
    CandidateEventType.SAFETY_CAR: OddEventType.RACE_CONTROL,
    CandidateEventType.YELLOW_FLAG: OddEventType.RACE_CONTROL,
    CandidateEventType.RED_FLAG: OddEventType.RACE_CONTROL,
}

ODD_TO_EVEN_EVENT_TYPE: Dict[OddEventType, CandidateEventType] = {
    OddEventType.OVERTAKE: CandidateEventType.OVERTAKE_OPPORTUNITY,
    OddEventType.LEAD_BATTLE: CandidateEventType.LEAD_BATTLE,
    OddEventType.PODIUM_BATTLE: CandidateEventType.ACTIVE_BATTLE,
    OddEventType.MIDFIELD_BATTLE: CandidateEventType.MIDFIELD_BATTLE,
    OddEventType.PIT_STRATEGY: CandidateEventType.PIT_STOP,
    OddEventType.RACE_CONTROL: CandidateEventType.SAFETY_CAR,
    OddEventType.INCIDENT: CandidateEventType.SAFETY_CAR,
    OddEventType.STRATEGIC_DEVELOPMENT: CandidateEventType.TIRE_DEGRADATION,
    OddEventType.TELEMETRY_CHANGE: CandidateEventType.DRS_THREAT,
}


# =========================================================================
# Helper utilities
# =========================================================================

def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def _session_key_from_info(info: OddSessionInfo) -> str:
    """Build an even-phase session_key string from odd-phase SessionInfo."""
    return f"{info.year}_{info.grand_prix}_{info.session_type}"


def _parse_session_clock(clock_str: Optional[str]) -> Optional[float]:
    """Convert 'HH:MM:SS' to seconds, or return None."""
    if not clock_str:
        return None
    parts = clock_str.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


def _weather_from_global(gs: OddGlobalState) -> Optional[str]:
    """Derive a compact weather string from odd GlobalState."""
    parts: list[str] = []
    if gs.rainfall:
        parts.append("Rain")
    if gs.air_temp_celsius is not None:
        parts.append(f"{gs.air_temp_celsius}°C")
    return ", ".join(parts) if parts else None


def _weather_from_race_global(gs: OddGlobalRaceStateSummary) -> Optional[str]:
    """Derive weather string from odd GlobalRaceStateSummary."""
    parts: list[str] = []
    if gs.rainfall:
        parts.append("Rain")
    if gs.air_temp_celsius is not None:
        parts.append(f"{gs.air_temp_celsius}°C")
    return ", ".join(parts) if parts else None


# =========================================================================
# 1. Phase 1 -> Phase 2: TelemetryFrame adapter
# =========================================================================

def odd_telemetry_to_even(odd_frame: OddTelemetryFrame) -> EvenTelemetryFrame:
    """Convert an odd-phase ``TelemetryFrame`` to an even-phase ``TelemetryFrame``.

    Key mappings:
    * ``abbreviation`` -> ``driver_id``
    * ``driver_number`` stored in metadata (dropped from even schema)
    * ``global_state`` flattened to ``flag_status``, ``weather``, ``session_clock``
    * ``session_info`` collapsed to a ``session_key`` string
    """
    drivers: list[EvenDriverState] = []
    for d in odd_frame.drivers:
        drivers.append(
            EvenDriverState(
                driver_id=d.abbreviation,
                position=d.position or 0,
                lap_number=d.lap_number or 0,
                gap_ahead=d.gap_ahead_sec,
                tire_compound=d.tire_compound,
                tire_age=d.tire_age_laps,
                drs_available=d.drs_active,
                speed=d.speed_kph,
                last_lap_time=None,
            )
        )

    gs = odd_frame.global_state
    return EvenTelemetryFrame(
        event_id=_new_id(),
        timestamp=odd_frame.timestamp,
        session_key=_session_key_from_info(odd_frame.session_info),
        lap=odd_frame.drivers[0].lap_number if odd_frame.drivers else 0,
        drivers=drivers,
        flag_status=gs.track_status,
        weather=_weather_from_global(gs),
        session_clock=_parse_session_clock(gs.session_clock),
    )


# =========================================================================
# 2. Phase 3 -> Phase 4: RaceStateSnapshot adapter
# =========================================================================

def odd_snapshot_to_even(odd_snapshot: OddRaceStateSnapshot) -> EvenRaceStateSnapshot:
    """Convert an odd-phase ``RaceStateSnapshot`` to even-phase ``RaceStateSnapshot``.

    Key mappings:
    * ``drivers`` dict -> ``drivers`` list
    * ``abbreviation`` -> ``driver_id``
    * ``gap_ahead_sec`` -> ``gap_ahead``
    * ``tire_age_laps`` -> ``tire_age``
    * ``drs_active`` -> ``drs_available``
    * ``degradation_score`` -> ``stint_deg_scores`` dict
    * ``relative_pace_delta`` -> ``pace_deltas`` dict
    * ``catch_time_laps`` -> ``catch_times`` dict
    * ``active_battles`` list -> ``battle_pairs`` list of tuples
    * ``global_state`` -> ``flag_status``, ``weather``
    """
    drivers: list[EvenDriverState] = []
    gaps: Dict[str, float] = {}
    pace_deltas: Dict[str, float] = {}
    catch_times: Dict[str, Optional[float]] = {}
    stint_deg_scores: Dict[str, float] = {}

    for abbr, d in odd_snapshot.drivers.items():
        drivers.append(
            EvenDriverState(
                driver_id=d.abbreviation,
                position=d.position or 0,
                lap_number=d.lap_number or 0,
                gap_ahead=d.gap_ahead_sec,
                tire_compound=d.tire_compound,
                tire_age=d.tire_age_laps,
                drs_available=d.drs_active,
                speed=d.speed_kph,
                last_lap_time=None,
            )
        )
        if d.gap_ahead_sec is not None:
            gaps[d.abbreviation] = d.gap_ahead_sec
        if d.relative_pace_delta is not None:
            pace_deltas[d.abbreviation] = d.relative_pace_delta
        if d.catch_time_laps is not None:
            catch_times[d.abbreviation] = d.catch_time_laps
        if d.degradation_score is not None:
            stint_deg_scores[d.abbreviation] = d.degradation_score

    battle_pairs: list[tuple[str, str]] = [
        (b.driver_ahead, b.driver_behind) for b in odd_snapshot.active_battles
    ]

    gs = odd_snapshot.global_state
    lap = 0
    if odd_snapshot.drivers:
        first = next(iter(odd_snapshot.drivers.values()))
        lap = first.lap_number or 0

    return EvenRaceStateSnapshot(
        event_id=_new_id(),
        timestamp=odd_snapshot.timestamp,
        session_key=_session_key_from_info(odd_snapshot.session_info),
        lap=lap,
        drivers=drivers,
        gaps=gaps,
        pace_deltas=pace_deltas,
        catch_times=catch_times,
        stint_deg_scores=stint_deg_scores,
        battle_pairs=battle_pairs,
        flag_status=gs.track_status,
        weather=_weather_from_race_global(gs),
    )


# =========================================================================
# 3. Phase 4 -> Phase 5: CandidateEvent adapter (even -> odd)
# =========================================================================

def even_candidate_to_odd(
    even_event: EvenCandidateEvent,
) -> OddCandidateEvent:
    """Convert an even-phase ``CandidateEvent`` to an odd-phase ``CandidateEvent``.

    Event-type mapping:
    * OVERTAKE_OPPORTUNITY, DRS_THREAT -> OVERTAKE
    * ACTIVE_BATTLE -> MIDFIELD_BATTLE  (or LEAD_BATTLE when severity >= 0.8)
    * LEAD_BATTLE -> LEAD_BATTLE
    * MIDFIELD_BATTLE -> MIDFIELD_BATTLE
    * PIT_STOP, PIT_STRATEGY_TRIGGER -> PIT_STRATEGY
    * TIRE_DEGRADATION -> STRATEGIC_DEVELOPMENT
    * SAFETY_CAR, YELLOW_FLAG, RED_FLAG -> RACE_CONTROL

    Other mappings:
    * ``explanation`` -> ``description``
    * ``severity`` >= 0.8 -> ``safety_critical = True``
    """
    # Special case: ACTIVE_BATTLE uses severity to decide target type
    if even_event.candidate_type == CandidateEventType.ACTIVE_BATTLE:
        odd_type = (
            OddEventType.LEAD_BATTLE
            if even_event.severity >= 0.8
            else OddEventType.MIDFIELD_BATTLE
        )
    else:
        odd_type = EVEN_TO_ODD_EVENT_TYPE.get(
            even_event.candidate_type, OddEventType.TELEMETRY_CHANGE
        )

    return OddCandidateEvent(
        event_id=even_event.event_id,
        timestamp=even_event.timestamp,
        event_type=odd_type,
        involved_drivers=even_event.involved_drivers,
        confidence=even_event.confidence,
        source_frame_id=0,
        description=even_event.explanation,
        safety_critical=even_event.severity >= 0.8,
    )


# =========================================================================
# 4. Phase 5 -> Phase 6: ScheduledBeat -> CandidateEvent adapter
# =========================================================================

def odd_beat_to_even_candidate(
    odd_beat: OddScheduledBeat,
) -> EvenCandidateEvent:
    """Extract ``CandidateEvent``-compatible data from an odd ``ScheduledBeat``
    for use in even-phase retrieval (Phase 6).

    Key mappings:
    * ``event_type`` -> ``candidate_type`` (reverse mapping)
    * ``priority_score`` -> ``confidence``
    * ``description`` -> ``explanation``
    * ``storyline_id`` passed through
    """
    candidate_type = ODD_TO_EVEN_EVENT_TYPE.get(
        odd_beat.event_type, CandidateEventType.ACTIVE_BATTLE
    )

    return EvenCandidateEvent(
        event_id=odd_beat.beat_id,
        timestamp=odd_beat.timestamp,
        candidate_type=candidate_type,
        involved_drivers=odd_beat.involved_drivers,
        confidence=min(max(odd_beat.priority_score, 0.0), 1.0),
        severity=0.5,
        explanation=odd_beat.description or "",
        storyline_id=odd_beat.storyline_id,
    )


# =========================================================================
# 5. Phase 9 -> Phase 10: GeneratedCommentary adapter (odd -> even)
# =========================================================================

def odd_commentary_to_even(
    odd_gen: OddGeneratedCommentary,
) -> EvenGeneratedCommentary:
    """Convert an odd-phase ``GeneratedCommentary`` to even-phase format.

    Key mappings:
    * ``beat_id`` -> ``source_beat_id``
    * ``commentary_text`` -> ``generated_text`` (post-processed version)
    * ``latency_ms`` -> ``generation_latency_ms``
    """
    return EvenGeneratedCommentary(
        event_id=_new_id(),
        timestamp=odd_gen.timestamp,
        source_beat_id=odd_gen.beat_id,
        prompt_text=odd_gen.prompt_text,
        generated_text=odd_gen.commentary_text or odd_gen.raw_output_text,
        model_variant=odd_gen.model_variant,
        generation_latency_ms=odd_gen.latency_ms,
        involved_drivers=odd_gen.involved_drivers,
    )


# =========================================================================
# 6. Phase 10 -> Phase 11: GuardedCommentary -> odd GeneratedCommentary
# =========================================================================

def even_guarded_to_odd_generated(
    guarded: EvenGuardedCommentary,
    original_odd_gen: OddGeneratedCommentary,
) -> OddGeneratedCommentary:
    """Merge even-phase guard adjustments back into the odd ``GeneratedCommentary``.

    Takes the original odd-phase ``GeneratedCommentary`` and replaces its
    ``commentary_text`` with the guard's ``adjusted_text``, preserving all
    other odd-phase fields.
    """
    data = original_odd_gen.model_dump()
    data["commentary_text"] = guarded.adjusted_text
    return OddGeneratedCommentary(**data)


# =========================================================================
# 7. Phase 11 -> Phase 12: FinalCommentary adapter (odd -> even)
# =========================================================================

def odd_final_to_even(odd_final: OddFinalCommentary) -> EvenFinalCommentary:
    """Convert an odd-phase ``FinalCommentary`` to even-phase format.

    Key mappings:
    * ``confidence_band`` enum -> string value
    * ``replay_lag_seconds`` -> ``replay_lag_ms``  (* 1000)
    * ``race_state_summary`` -> dict
    """
    # Convert OddConfidenceBand enum to plain string for even side
    confidence_str: str = (
        odd_final.confidence_band.value
        if isinstance(odd_final.confidence_band, OddConfidenceBand)
        else str(odd_final.confidence_band)
    )

    # Convert race_state_summary to dict
    race_summary_dict: dict = {}
    if odd_final.race_state_summary is not None:
        race_summary_dict = odd_final.race_state_summary.model_dump()

    return EvenFinalCommentary(
        event_id=_new_id(),
        timestamp=odd_final.timestamp,
        storyline_id=odd_final.storyline_id,
        final_text=odd_final.final_text,
        confidence_band=confidence_str,
        replay_lag_ms=odd_final.replay_lag_seconds * 1000.0,
        race_state_summary=race_summary_dict,
        involved_drivers=odd_final.involved_drivers,
    )


# =========================================================================
# 8. Phase 13: BaselineCommentary adapter (odd -> even GeneratedCommentary)
# =========================================================================

def odd_baseline_to_even(odd_baseline: OddBaselineCommentary) -> EvenGeneratedCommentary:
    """Convert an odd-phase ``BaselineCommentary`` to even-phase ``GeneratedCommentary``.

    Baseline outputs share enough structure with generated commentary
    to be mapped into the same even-phase type for unified evaluation.
    """
    # Map odd EventType to even CandidateEventType
    odd_type_name = (
        odd_baseline.event_type.value
        if isinstance(odd_baseline.event_type, OddEventType)
        else str(odd_baseline.event_type)
    )
    even_type = ODD_TO_EVEN_EVENT_TYPE.get(odd_type_name, CandidateEventType.TELEMETRY_CHANGE)

    return EvenGeneratedCommentary(
        event_id=_new_id(),
        timestamp=odd_baseline.timestamp,
        source_beat_id=odd_baseline.beat_id,
        candidate_type=even_type,
        generated_text=odd_baseline.commentary_text or odd_baseline.raw_output_text,
        model_name=odd_baseline.model_name,
        latency_ms=odd_baseline.latency_ms,
        involved_drivers=odd_baseline.involved_drivers,
    )
