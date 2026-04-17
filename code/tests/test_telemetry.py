"""Tests for schema validity, chronological ordering, replay speed, and serialization."""

from __future__ import annotations

import json
import time
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    GlobalState,
    SessionInfo,
    TelemetryFrame,
)
from src.telemetry_to_narrative.adapters.replay_adapter import ReplayAdapter


# ---------------------------------------------------------------------------
# Helpers to build a mock FastF1 session
# ---------------------------------------------------------------------------


def _make_mock_session(n_laps: int = 20, n_drivers: int = 3) -> MagicMock:
    """Create a mock FastF1 session with synthetic lap + weather data."""
    session = MagicMock()

    base_time = pd.Timestamp("2024-09-01 14:00:00", tz="UTC")
    lap_rows = []

    for d in range(n_drivers):
        driver_num = str(10 + d)
        abbrev = ["VER", "HAM", "LEC"][d % 3]
        cumulative = timedelta(0)

        for lap in range(n_laps):
            lap_duration = timedelta(seconds=80 + d * 0.5 + np.random.uniform(-1, 1))
            cumulative += lap_duration
            start = base_time + timedelta(seconds=lap * 85 + d * 2)
            lap_rows.append(
                {
                    "DriverNumber": driver_num,
                    "Driver": abbrev,
                    "LapNumber": lap + 1,
                    "LapStartDate": start,
                    "LapTime": lap_duration,
                    "Time": cumulative,
                    "Position": d + 1,
                    "Compound": ["SOFT", "MEDIUM", "HARD"][lap % 3],
                    "TyreLife": (lap % 15) + 1,
                    "SpeedST": 310.0 + np.random.uniform(-5, 5),
                }
            )

    laps_df = pd.DataFrame(lap_rows)
    session.laps = laps_df

    # Event metadata
    session.event = {
        "EventName": "Italian Grand Prix",
        "EventDate": pd.Timestamp("2024-09-01"),
    }
    session.name = "Race"

    # Weather data
    weather_rows = []
    for i in range(10):
        weather_rows.append(
            {
                "Time": pd.Timedelta(minutes=i * 10),
                "AirTemp": 28.0 + i * 0.2,
                "TrackTemp": 42.0 + i * 0.3,
                "Humidity": 45.0,
                "Rainfall": 0,
                "WindSpeed": 2.5,
            }
        )
    session.weather_data = pd.DataFrame(weather_rows)

    # Race control messages
    session.race_control_messages = pd.DataFrame(
        [
            {
                "Time": pd.Timedelta(minutes=5),
                "Category": "Flag",
                "Message": "GREEN LIGHT",
            },
            {
                "Time": pd.Timedelta(minutes=30),
                "Category": "SafetyCar",
                "Message": "SAFETY CAR DEPLOYED",
            },
        ]
    )

    return session


# ---------------------------------------------------------------------------
# 1. Schema validity
# ---------------------------------------------------------------------------


class TestSchemaValidity:
    def test_driver_state_minimal(self):
        ds = DriverState(driver_number="1", abbreviation="VER")
        assert ds.driver_number == "1"
        assert ds.position is None

    def test_driver_state_full(self):
        ds = DriverState(
            driver_number="44",
            abbreviation="HAM",
            position=2,
            lap_number=15,
            gap_ahead_sec=1.234,
            tire_compound="MEDIUM",
            tire_age_laps=8,
            drs_active=True,
            speed_kph=320.5,
        )
        assert ds.position == 2
        assert ds.drs_active is True

    def test_global_state_defaults(self):
        gs = GlobalState()
        assert gs.track_status is None
        assert gs.rainfall is None

    def test_telemetry_frame_construction(self):
        frame = TelemetryFrame(
            frame_id=0,
            timestamp=datetime.now(timezone.utc),
            session_info=SessionInfo(year=2024, grand_prix="Monza", session_type="R"),
        )
        assert frame.frame_id == 0
        assert frame.drivers == []

    def test_frame_id_must_be_non_negative(self):
        with pytest.raises(Exception):
            TelemetryFrame(
                frame_id=-1,
                timestamp=datetime.now(timezone.utc),
                session_info=SessionInfo(year=2024, grand_prix="Monza", session_type="R"),
            )


# ---------------------------------------------------------------------------
# 2. Chronological ordering
# ---------------------------------------------------------------------------


class TestChronologicalOrdering:
    def test_frames_are_chronologically_ordered(self):
        session = _make_mock_session(n_laps=10, n_drivers=2)
        adapter = ReplayAdapter(session, speed=10.0)

        frames = list(adapter.replay(realtime=False))
        assert len(frames) > 1

        timestamps = [f.timestamp for f in frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Frame {i} timestamp {timestamps[i]} is before frame {i-1} "
                f"timestamp {timestamps[i-1]}"
            )

    def test_frame_ids_are_monotonic(self):
        session = _make_mock_session(n_laps=5, n_drivers=2)
        adapter = ReplayAdapter(session, speed=10.0)

        frames = list(adapter.replay(realtime=False))
        ids = [f.frame_id for f in frames]
        assert ids == list(range(len(ids)))


# ---------------------------------------------------------------------------
# 3. Replay speed control
# ---------------------------------------------------------------------------


class TestReplaySpeed:
    def test_fast_replay_no_sleeping(self):
        session = _make_mock_session(n_laps=5, n_drivers=1)
        adapter = ReplayAdapter(session, speed=100.0)

        t0 = time.monotonic()
        frames = list(adapter.replay(max_frames=10, realtime=True))
        elapsed = time.monotonic() - t0

        # At 100x speed with 1s interval, sleep is 0.01s per frame → ~0.1s for 10
        assert elapsed < 2.0, f"Fast replay took too long: {elapsed:.2f}s"

    def test_nonrealtime_is_instant(self):
        session = _make_mock_session(n_laps=5, n_drivers=1)
        adapter = ReplayAdapter(session, speed=1.0)

        t0 = time.monotonic()
        frames = list(adapter.replay(max_frames=10, realtime=False))
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"Non-realtime replay should be instant, took {elapsed:.2f}s"

    def test_max_frames_limits_output(self):
        session = _make_mock_session(n_laps=20, n_drivers=2)
        adapter = ReplayAdapter(session, speed=100.0)

        frames = list(adapter.replay(max_frames=5, realtime=False))
        assert len(frames) == 5


# ---------------------------------------------------------------------------
# 4. Frame serialization (JSON / JSONL)
# ---------------------------------------------------------------------------


class TestFrameSerialization:
    def test_frame_round_trips_json(self):
        frame = TelemetryFrame(
            frame_id=42,
            timestamp=datetime(2024, 9, 1, 14, 30, 0, tzinfo=timezone.utc),
            session_time=timedelta(minutes=30),
            session_info=SessionInfo(year=2024, grand_prix="Monza", session_type="R"),
            drivers=[
                DriverState(
                    driver_number="1",
                    abbreviation="VER",
                    position=1,
                    lap_number=15,
                    speed_kph=318.2,
                )
            ],
            global_state=GlobalState(air_temp_celsius=28.5, rainfall=False),
        )
        json_str = frame.model_dump_json()
        restored = TelemetryFrame.model_validate_json(json_str)

        assert restored.frame_id == 42
        assert restored.drivers[0].abbreviation == "VER"
        assert restored.global_state.air_temp_celsius == 28.5

    def test_jsonl_output_file(self):
        session = _make_mock_session(n_laps=5, n_drivers=2)
        adapter = ReplayAdapter(session, speed=10.0)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out_path = Path(tmp.name)

        adapter.to_jsonl(out_path, max_frames=10)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) <= 10
        assert len(lines) > 0

        # Each line should be valid JSON and deserialize to a TelemetryFrame
        for line in lines:
            data = json.loads(line)
            frame = TelemetryFrame.model_validate(data)
            assert frame.session_info.grand_prix == "Italian Grand Prix"

        out_path.unlink()

    def test_jsonl_lines_are_chronological(self):
        session = _make_mock_session(n_laps=10, n_drivers=2)
        adapter = ReplayAdapter(session, speed=10.0)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out_path = Path(tmp.name)

        adapter.to_jsonl(out_path)

        lines = out_path.read_text().strip().split("\n")
        timestamps = []
        for line in lines:
            frame = TelemetryFrame.model_validate_json(line)
            timestamps.append(frame.timestamp)

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

        out_path.unlink()
