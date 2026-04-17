"""Generate sample JSONL output using synthetic data (no FastF1 network needed)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    GlobalState,
    SessionInfo,
    TelemetryFrame,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "logs"


def generate_sample_frames(n_frames: int = 20) -> list[TelemetryFrame]:
    """Build synthetic frames that mirror real FastF1 output structure."""
    session_info = SessionInfo(year=2024, grand_prix="Italian Grand Prix", session_type="Race")
    base_ts = datetime(2024, 9, 1, 14, 0, 0, tzinfo=timezone.utc)

    drivers_template = [
        ("1", "VER", "MEDIUM"),
        ("11", "PER", "HARD"),
        ("44", "HAM", "SOFT"),
        ("63", "RUS", "MEDIUM"),
        ("16", "LEC", "SOFT"),
        ("55", "SAI", "HARD"),
    ]

    frames: list[TelemetryFrame] = []
    for i in range(n_frames):
        elapsed = timedelta(seconds=i * 85)
        lap = i + 1
        drivers = []
        for pos, (num, abbr, compound) in enumerate(drivers_template, start=1):
            gap = round(0.3 * pos + (i % 3) * 0.1, 3) if pos > 1 else None
            ds = DriverState(
                driver_number=num,
                abbreviation=abbr,
                position=pos,
                lap_number=lap,
                gap_ahead_sec=gap,
                tire_compound=compound,
                tire_age_laps=lap if lap <= 15 else lap - 15,
                drs_active=(pos > 1 and gap is not None and gap < 1.0),
                speed_kph=round(305 + pos * 1.2 + (i % 5) * 0.5, 1),
            )
            drivers.append(ds)

        weather_flag = "GREEN FLAG"
        sc = False
        if 8 <= i <= 10:
            weather_flag = "SAFETY CAR DEPLOYED"
            sc = True

        gs = GlobalState(
            track_status=weather_flag,
            safety_car=sc,
            air_temp_celsius=round(28.0 + i * 0.05, 1),
            track_temp_celsius=round(42.0 + i * 0.1, 1),
            humidity_pct=45.0,
            rainfall=False,
            wind_speed_ms=2.3,
            session_clock=str(elapsed),
        )

        frame = TelemetryFrame(
            frame_id=i,
            timestamp=base_ts + elapsed,
            session_time=elapsed,
            session_info=session_info,
            drivers=drivers,
            global_state=gs,
        )
        frames.append(frame)

    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sample_replay.jsonl"

    frames = generate_sample_frames(20)
    with open(out_path, "w") as fh:
        for frame in frames:
            fh.write(frame.model_dump_json() + "\n")

    print(f"Wrote {len(frames)} sample frames to {out_path}")


if __name__ == "__main__":
    main()
