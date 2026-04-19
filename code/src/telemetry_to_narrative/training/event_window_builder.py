"""EventWindow extraction — build context windows around selected beats.

Given a sequence of RaceStateSnapshots and the ScheduledBeats produced by
the editorial scheduler, this module extracts local context windows that
pair the beat metadata with the surrounding snapshot state.  Each window
becomes the structured input for one SFT example.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.sft_dataset import EventWindow

logger = logging.getLogger(__name__)

# Default half-width of the context window (in frames).
DEFAULT_CONTEXT_RADIUS = 2


def _make_window_id() -> str:
    return uuid.uuid4().hex[:12]


def _build_state_summary(snapshot: RaceStateSnapshot, involved_drivers: list[str]) -> dict:
    """Build a compact state summary dict for the centre frame.

    Includes global state, involved drivers' state, and derived summary
    fields that will be useful for constructing SFT input text.
    """
    summary: dict = {
        "frame_id": snapshot.frame_id,
        "lap": None,
        "track_status": snapshot.global_state.track_status,
        "safety_car": snapshot.global_state.safety_car,
        "leader": snapshot.derived_summary.leader,
        "total_drivers": snapshot.derived_summary.total_drivers,
        "active_battle_count": snapshot.derived_summary.active_battles,
    }

    # Add lap number from leader or first available driver.
    for d in snapshot.drivers.values():
        if d.lap_number is not None:
            summary["lap"] = d.lap_number
            break

    # Per-driver state for involved drivers.
    driver_states = {}
    for abbr in involved_drivers:
        ds = snapshot.drivers.get(abbr)
        if ds is None:
            continue
        driver_states[abbr] = {
            "position": ds.position,
            "lap_number": ds.lap_number,
            "gap_ahead_sec": ds.gap_ahead_sec,
            "tire_compound": ds.tire_compound,
            "tire_age_laps": ds.tire_age_laps,
            "speed_kph": ds.speed_kph,
            "pace_trend": ds.pace_trend,
            "degradation_score": ds.degradation_score,
            "pit_window_open": ds.pit_window_open,
        }
    summary["drivers"] = driver_states

    # Active battles involving these drivers.
    battles = []
    involved_set = set(involved_drivers)
    for b in snapshot.active_battles:
        if b.driver_ahead in involved_set or b.driver_behind in involved_set:
            battles.append({
                "ahead": b.driver_ahead,
                "behind": b.driver_behind,
                "gap_sec": b.gap_sec,
                "intensity": b.battle_intensity,
            })
    summary["battles"] = battles

    return summary


def extract_event_windows(
    snapshots: list[RaceStateSnapshot],
    beats: list[ScheduledBeat],
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
    selected_only: bool = True,
) -> list[EventWindow]:
    """Build EventWindows from snapshots and scheduled beats.

    Parameters
    ----------
    snapshots : list[RaceStateSnapshot]
        Ordered sequence of snapshots (by frame_id).
    beats : list[ScheduledBeat]
        Beats produced by the editorial scheduler.
    context_radius : int
        Number of frames on each side of the centre to include.
    selected_only : bool
        If True, only build windows for beats with ``selected_for_generation=True``.

    Returns
    -------
    list[EventWindow]
        One window per qualifying beat, sorted by centre timestamp.
    """
    if not snapshots:
        return []

    # Build frame_id → index lookup.
    frame_index: dict[int, int] = {s.frame_id: i for i, s in enumerate(snapshots)}

    # Filter beats.
    target_beats = [b for b in beats if (not selected_only or b.selected_for_generation)]

    windows: list[EventWindow] = []
    for beat in target_beats:
        centre_idx = frame_index.get(beat.source_frame_id)
        if centre_idx is None:
            logger.warning(
                "Beat %s references frame %d which is not in the snapshot list; skipping.",
                beat.beat_id, beat.source_frame_id,
            )
            continue

        # Determine window bounds (clamped to available range).
        lo = max(0, centre_idx - context_radius)
        hi = min(len(snapshots) - 1, centre_idx + context_radius)

        window_snaps = snapshots[lo: hi + 1]
        centre_snap = snapshots[centre_idx]

        state_summary = _build_state_summary(centre_snap, beat.involved_drivers)

        window = EventWindow(
            window_id=_make_window_id(),
            frame_ids=[s.frame_id for s in window_snaps],
            timestamps=[s.timestamp for s in window_snaps],
            centre_frame_id=centre_snap.frame_id,
            centre_timestamp=centre_snap.timestamp,
            session_info=centre_snap.session_info,
            event_type=beat.event_type,
            storyline_id=beat.storyline_id,
            involved_drivers=beat.involved_drivers,
            priority_score=beat.priority_score,
            beat_description=beat.description,
            state_summary=state_summary,
        )
        windows.append(window)

    windows.sort(key=lambda w: w.centre_timestamp)
    logger.info(
        "Extracted %d event windows from %d beats (%d snapshots).",
        len(windows), len(target_beats), len(snapshots),
    )
    return windows
