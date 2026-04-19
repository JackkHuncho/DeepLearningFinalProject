"""Debug and inspection utilities for RaceStateSnapshots.

Provides:
- Pretty-print current race state for selected drivers
- Save snapshots to JSONL or CSV
- Compare consecutive snapshots and report deltas
"""

from __future__ import annotations

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Optional, Sequence

from src.telemetry_to_narrative.schemas.race_state import (
    RaceStateSnapshot,
    DriverRaceState,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pretty-print
# ------------------------------------------------------------------


def format_driver_state(drs: DriverRaceState) -> str:
    """Human-readable single-driver summary."""
    parts = [f"P{drs.position or '?':>2}  {drs.abbreviation}"]

    if drs.lap_number is not None:
        parts.append(f"Lap {drs.lap_number}")

    if drs.gap_ahead_sec is not None:
        parts.append(f"Gap +{drs.gap_ahead_sec:.3f}s")

    if drs.tire_compound:
        age_str = f"{drs.tire_age_laps}L" if drs.tire_age_laps is not None else "?"
        parts.append(f"{drs.tire_compound}({age_str})")

    if drs.recent_pace_sec is not None:
        parts.append(f"Pace {drs.recent_pace_sec:.2f}s")

    if drs.pace_trend is not None:
        direction = "+" if drs.pace_trend >= 0 else ""
        parts.append(f"Trend {direction}{drs.pace_trend:.3f}")

    if drs.catch_time_laps is not None:
        parts.append(f"Catch ~{drs.catch_time_laps:.1f}L")

    if drs.degradation_score is not None:
        parts.append(f"Deg {drs.degradation_score:+.4f}")

    if drs.pit_window_open is True:
        parts.append("PIT-WINDOW")

    if drs.drs_active:
        parts.append("DRS")

    return "  |  ".join(parts)


def print_snapshot(
    snapshot: RaceStateSnapshot,
    drivers: Optional[Sequence[str]] = None,
) -> str:
    """Format a snapshot as a human-readable string.

    Parameters
    ----------
    snapshot : RaceStateSnapshot
    drivers : list[str] | None
        If provided, only show these driver abbreviations.
        Otherwise show all drivers.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    lines: list[str] = []
    lines.append(
        f"=== Frame {snapshot.frame_id}  "
        f"{snapshot.timestamp.isoformat()}  "
        f"{snapshot.session_info.grand_prix} {snapshot.session_info.session_type} ==="
    )

    gs = snapshot.global_state
    status_parts = []
    if gs.track_status:
        status_parts.append(gs.track_status)
    if gs.safety_car:
        status_parts.append("SC")
    if gs.rainfall:
        status_parts.append("RAIN")
    if gs.air_temp_celsius is not None:
        status_parts.append(f"Air {gs.air_temp_celsius}°C")
    if gs.track_temp_celsius is not None:
        status_parts.append(f"Track {gs.track_temp_celsius}°C")
    if status_parts:
        lines.append("  Status: " + " | ".join(status_parts))

    # Filter drivers.
    driver_list = list(snapshot.drivers.values())
    driver_list.sort(key=lambda d: d.position if d.position is not None else 999)

    if drivers:
        abbr_set = {d.upper() for d in drivers}
        driver_list = [d for d in driver_list if d.abbreviation in abbr_set]

    for drs in driver_list:
        lines.append("  " + format_driver_state(drs))

    if snapshot.active_battles:
        lines.append("  --- Battles ---")
        for b in snapshot.active_battles:
            lines.append(
                f"    {b.driver_behind} -> {b.driver_ahead}  "
                f"gap={b.gap_sec}s  intensity={b.battle_intensity:.3f}  "
                f"dur={b.battle_duration_laps}L"
            )

    ds = snapshot.derived_summary
    lines.append(
        f"  Summary: leader={ds.leader}  drivers={ds.total_drivers}  "
        f"battles={ds.active_battles}  SC={ds.safety_car_active}"
    )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def save_snapshots_jsonl(
    snapshots: Sequence[RaceStateSnapshot],
    path: Path | str,
) -> Path:
    """Write snapshots to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for snap in snapshots:
            fh.write(snap.model_dump_json() + "\n")
    logger.info("Wrote %d snapshots to %s", len(snapshots), p)
    return p


def save_snapshots_csv(
    snapshots: Sequence[RaceStateSnapshot],
    path: Path | str,
) -> Path:
    """Write a flattened per-driver CSV of snapshot data.

    Each row is (frame_id, timestamp, driver_abbr, position, gap_ahead, ...).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "timestamp",
        "driver",
        "position",
        "lap_number",
        "gap_ahead_sec",
        "gap_behind_sec",
        "tire_compound",
        "tire_age_laps",
        "stint_length",
        "drs_active",
        "speed_kph",
        "recent_pace_sec",
        "pace_trend",
        "relative_pace_delta",
        "catch_time_laps",
        "degradation_score",
        "pit_window_open",
        "track_status",
        "safety_car",
    ]

    with open(p, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for snap in snapshots:
            for abbr, drs in snap.drivers.items():
                writer.writerow(
                    {
                        "frame_id": snap.frame_id,
                        "timestamp": snap.timestamp.isoformat(),
                        "driver": abbr,
                        "position": drs.position,
                        "lap_number": drs.lap_number,
                        "gap_ahead_sec": drs.gap_ahead_sec,
                        "gap_behind_sec": drs.gap_behind_sec,
                        "tire_compound": drs.tire_compound,
                        "tire_age_laps": drs.tire_age_laps,
                        "stint_length": drs.stint_length,
                        "drs_active": drs.drs_active,
                        "speed_kph": drs.speed_kph,
                        "recent_pace_sec": drs.recent_pace_sec,
                        "pace_trend": drs.pace_trend,
                        "relative_pace_delta": drs.relative_pace_delta,
                        "catch_time_laps": drs.catch_time_laps,
                        "degradation_score": drs.degradation_score,
                        "pit_window_open": drs.pit_window_open,
                        "track_status": snap.global_state.track_status,
                        "safety_car": snap.global_state.safety_car,
                    }
                )

    logger.info("Wrote %d snapshots to %s", len(snapshots), p)
    return p


# ------------------------------------------------------------------
# Snapshot comparison
# ------------------------------------------------------------------


def compare_snapshots(
    prev: RaceStateSnapshot,
    curr: RaceStateSnapshot,
) -> dict:
    """Compare two consecutive snapshots, returning a dict of deltas.

    Useful for debugging state transitions frame-by-frame.
    """
    deltas: dict = {
        "frame_id_from": prev.frame_id,
        "frame_id_to": curr.frame_id,
        "position_changes": [],
        "new_battles": [],
        "ended_battles": [],
        "pit_stops": [],
        "flag_change": None,
    }

    # Position changes.
    for abbr, curr_d in curr.drivers.items():
        prev_d = prev.drivers.get(abbr)
        if prev_d is None:
            continue
        if prev_d.position != curr_d.position and prev_d.position is not None and curr_d.position is not None:
            deltas["position_changes"].append(
                {
                    "driver": abbr,
                    "from": prev_d.position,
                    "to": curr_d.position,
                    "direction": "gained" if curr_d.position < prev_d.position else "lost",
                }
            )

    # Battle diff.
    prev_pairs = {(b.driver_ahead, b.driver_behind) for b in prev.active_battles}
    curr_pairs = {(b.driver_ahead, b.driver_behind) for b in curr.active_battles}
    deltas["new_battles"] = [
        {"ahead": a, "behind": b} for a, b in (curr_pairs - prev_pairs)
    ]
    deltas["ended_battles"] = [
        {"ahead": a, "behind": b} for a, b in (prev_pairs - curr_pairs)
    ]

    # Pit stops (compound change).
    for abbr, curr_d in curr.drivers.items():
        prev_d = prev.drivers.get(abbr)
        if prev_d is None:
            continue
        if (
            prev_d.tire_compound is not None
            and curr_d.tire_compound is not None
            and prev_d.tire_compound != curr_d.tire_compound
        ):
            deltas["pit_stops"].append(
                {
                    "driver": abbr,
                    "from_compound": prev_d.tire_compound,
                    "to_compound": curr_d.tire_compound,
                }
            )

    # Flag change.
    if prev.global_state.track_status != curr.global_state.track_status:
        deltas["flag_change"] = {
            "from": prev.global_state.track_status,
            "to": curr.global_state.track_status,
        }

    return deltas
