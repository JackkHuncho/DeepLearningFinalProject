"""Temporary adapter: RaceStateSnapshot → list[CandidateEvent].

This module bridges Phase 3 output directly to the Phase 5 scheduler,
bypassing the Phase 4 event-recognition pipeline.  When Phase 4 is ready,
this adapter can be removed and the scheduler fed real CandidateEvent
objects from the event bus.

The adapter applies simple heuristics to extract event candidates:
- Safety-car / race-control events from global state
- Lead and podium battles from active_battles
- Midfield battles
- Pit stops (detected via compound changes between snapshots)
- Telemetry changes as low-priority filler
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.candidate_event import (
    CandidateEvent,
    EventType,
)


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


def extract_candidates(
    snapshot: RaceStateSnapshot,
    prev_snapshot: Optional[RaceStateSnapshot] = None,
) -> list[CandidateEvent]:
    """Derive CandidateEvents from a RaceStateSnapshot.

    Parameters
    ----------
    snapshot : RaceStateSnapshot
        Current frame's state.
    prev_snapshot : RaceStateSnapshot | None
        Previous frame's state (used for change detection).

    Returns
    -------
    list[CandidateEvent]
        Candidate events extracted from this snapshot.
    """
    candidates: list[CandidateEvent] = []

    # 1. Race-control events.
    candidates.extend(_extract_race_control(snapshot, prev_snapshot))

    # 2. Battle events from active_battles.
    candidates.extend(_extract_battles(snapshot))

    # 3. Pit-stop events (compound change detection).
    if prev_snapshot is not None:
        candidates.extend(_extract_pit_stops(snapshot, prev_snapshot))

    # 4. Position changes (overtakes).
    if prev_snapshot is not None:
        candidates.extend(_extract_overtakes(snapshot, prev_snapshot))

    # 5. If no events were found, emit a low-priority telemetry-change filler.
    if not candidates:
        leader = snapshot.derived_summary.leader
        candidates.append(
            CandidateEvent(
                event_id=_make_id(),
                timestamp=snapshot.timestamp,
                event_type=EventType.TELEMETRY_CHANGE,
                involved_drivers=[leader] if leader else [],
                confidence=0.5,
                source_frame_id=snapshot.frame_id,
                description="Routine telemetry update",
                position_of_interest=1,
            )
        )

    return candidates


# ── Extraction helpers ────────────────────────────────────────────────────

def _extract_race_control(
    snap: RaceStateSnapshot,
    prev: Optional[RaceStateSnapshot],
) -> list[CandidateEvent]:
    """Detect safety-car, VSC, red-flag transitions."""
    events: list[CandidateEvent] = []
    gs = snap.global_state

    # Only emit if the track status changed from the previous snapshot.
    prev_status = prev.global_state.track_status if prev else None
    if gs.track_status and gs.track_status != prev_status:
        is_safety = bool(gs.safety_car) or bool(gs.red_flag) or bool(gs.virtual_safety_car)
        events.append(
            CandidateEvent(
                event_id=_make_id(),
                timestamp=snap.timestamp,
                event_type=EventType.RACE_CONTROL,
                confidence=1.0,
                source_frame_id=snap.frame_id,
                description=f"Race control: {gs.track_status}",
                safety_critical=is_safety,
            )
        )
    return events


def _extract_battles(snap: RaceStateSnapshot) -> list[CandidateEvent]:
    """Convert active battles into candidate events."""
    events: list[CandidateEvent] = []
    for battle in snap.active_battles:
        # Classify by position of the leading driver.
        ahead_state = snap.drivers.get(battle.driver_ahead)
        pos = ahead_state.position if ahead_state else None

        if pos == 1:
            etype = EventType.LEAD_BATTLE
        elif pos is not None and pos <= 3:
            etype = EventType.PODIUM_BATTLE
        else:
            etype = EventType.MIDFIELD_BATTLE

        events.append(
            CandidateEvent(
                event_id=_make_id(),
                timestamp=snap.timestamp,
                event_type=etype,
                involved_drivers=[battle.driver_ahead, battle.driver_behind],
                confidence=min(1.0, (battle.battle_intensity or 0.5) + 0.3),
                source_frame_id=snap.frame_id,
                description=(
                    f"{battle.driver_behind} chasing {battle.driver_ahead} "
                    f"(gap {battle.gap_sec}s, intensity {battle.battle_intensity})"
                ),
                position_of_interest=pos,
                gap_sec=battle.gap_sec,
                battle_intensity=battle.battle_intensity,
            )
        )
    return events


def _extract_pit_stops(
    snap: RaceStateSnapshot,
    prev: RaceStateSnapshot,
) -> list[CandidateEvent]:
    """Detect compound changes between consecutive snapshots."""
    events: list[CandidateEvent] = []
    for abbr, curr_d in snap.drivers.items():
        prev_d = prev.drivers.get(abbr)
        if prev_d is None:
            continue
        if (
            prev_d.tire_compound is not None
            and curr_d.tire_compound is not None
            and prev_d.tire_compound != curr_d.tire_compound
        ):
            events.append(
                CandidateEvent(
                    event_id=_make_id(),
                    timestamp=snap.timestamp,
                    event_type=EventType.PIT_STRATEGY,
                    involved_drivers=[abbr],
                    confidence=1.0,
                    source_frame_id=snap.frame_id,
                    description=(
                        f"{abbr} pitted: {prev_d.tire_compound} → {curr_d.tire_compound}"
                    ),
                    position_of_interest=curr_d.position,
                )
            )
    return events


def _extract_overtakes(
    snap: RaceStateSnapshot,
    prev: RaceStateSnapshot,
) -> list[CandidateEvent]:
    """Detect position gains between consecutive snapshots."""
    events: list[CandidateEvent] = []
    for abbr, curr_d in snap.drivers.items():
        prev_d = prev.drivers.get(abbr)
        if prev_d is None:
            continue
        if (
            prev_d.position is not None
            and curr_d.position is not None
            and curr_d.position < prev_d.position
        ):
            events.append(
                CandidateEvent(
                    event_id=_make_id(),
                    timestamp=snap.timestamp,
                    event_type=EventType.OVERTAKE,
                    involved_drivers=[abbr],
                    confidence=0.9,
                    source_frame_id=snap.frame_id,
                    description=(
                        f"{abbr} gained position: P{prev_d.position} → P{curr_d.position}"
                    ),
                    position_of_interest=curr_d.position,
                )
            )
    return events
