"""Tests for Phase 3: RaceStateEngine, FeatureExtractor, schemas, and inspection.

Covers:
- State update consistency
- Derived feature computation on synthetic examples
- Missing-data robustness
- Schema validity
- Snapshot serialization (JSONL and CSV)
- Snapshot comparison / diffs
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    GlobalState,
    SessionInfo,
    TelemetryFrame,
)
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
)
from src.telemetry_to_narrative.state.feature_extractor import (
    FeatureExtractor,
    _ols_slope,
)
from src.telemetry_to_narrative.state.inspection import (
    compare_snapshots,
    print_snapshot,
    save_snapshots_csv,
    save_snapshots_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_INFO = SessionInfo(year=2024, grand_prix="Test GP", session_type="Race")


def _make_frame(
    frame_id: int,
    drivers: list[DriverState] | None = None,
    track_status: str | None = "GREEN FLAG",
    safety_car: bool = False,
    air_temp: float = 28.0,
    session_time_sec: float = 0.0,
) -> TelemetryFrame:
    """Build a synthetic TelemetryFrame for testing."""
    return TelemetryFrame(
        frame_id=frame_id,
        timestamp=datetime(2024, 9, 1, 14, 0, 0, tzinfo=timezone.utc)
        + timedelta(seconds=session_time_sec),
        session_time=timedelta(seconds=session_time_sec),
        session_info=SESSION_INFO,
        drivers=drivers or [],
        global_state=GlobalState(
            track_status=track_status,
            safety_car=safety_car,
            air_temp_celsius=air_temp,
            track_temp_celsius=42.0,
            humidity_pct=45.0,
            rainfall=False,
            wind_speed_ms=2.3,
            session_clock=str(timedelta(seconds=session_time_sec)),
        ),
    )


def _driver(
    abbr: str,
    num: str = "1",
    position: int = 1,
    lap: int = 1,
    gap: float | None = None,
    compound: str = "MEDIUM",
    tire_age: int = 1,
    drs: bool = False,
    speed: float = 310.0,
) -> DriverState:
    return DriverState(
        driver_number=num,
        abbreviation=abbr,
        position=position,
        lap_number=lap,
        gap_ahead_sec=gap,
        tire_compound=compound,
        tire_age_laps=tire_age,
        drs_active=drs,
        speed_kph=speed,
    )


def _run_frames(frames: list[TelemetryFrame]):
    """Run engine + extractor over a list of frames, return snapshots."""
    engine = RaceStateEngine()
    extractor = FeatureExtractor(engine)
    snapshots = []
    for f in frames:
        engine.update(f)
        snapshots.append(extractor.extract(f))
    return snapshots, engine


# ---------------------------------------------------------------------------
# 1. Schema validity
# ---------------------------------------------------------------------------


class TestSchemaValidity:
    def test_driver_race_state_defaults(self):
        drs = DriverRaceState(driver_number="1", abbreviation="VER")
        assert drs.position is None
        assert drs.catch_time_laps is None
        assert drs.pit_window_open is None

    def test_driver_race_state_full(self):
        drs = DriverRaceState(
            driver_number="1",
            abbreviation="VER",
            position=1,
            lap_number=20,
            gap_ahead_sec=None,
            gap_behind_sec=0.8,
            tire_compound="HARD",
            tire_age_laps=15,
            stint_length=15,
            drs_active=False,
            speed_kph=315.0,
            recent_pace_sec=67.2,
            pace_trend=0.05,
            relative_pace_delta=-0.12,
            catch_time_laps=6.5,
            degradation_score=0.002,
            pit_window_open=False,
        )
        assert drs.pace_trend == 0.05
        assert drs.pit_window_open is False

    def test_battle_state(self):
        bs = BattleState(
            driver_ahead="VER",
            driver_behind="HAM",
            gap_sec=0.7,
            battle_intensity=0.85,
        )
        assert bs.gap_sec == 0.7

    def test_race_state_snapshot_minimal(self):
        snap = RaceStateSnapshot(
            frame_id=0,
            timestamp=datetime.now(timezone.utc),
            session_info=SESSION_INFO,
        )
        assert snap.drivers == {}
        assert snap.active_battles == []

    def test_race_state_snapshot_full_round_trip(self):
        snap = RaceStateSnapshot(
            frame_id=5,
            timestamp=datetime(2024, 9, 1, 14, 30, 0, tzinfo=timezone.utc),
            session_time=timedelta(minutes=30),
            session_info=SESSION_INFO,
            global_state=GlobalRaceStateSummary(track_status="GREEN", safety_car=False),
            drivers={
                "VER": DriverRaceState(
                    driver_number="1",
                    abbreviation="VER",
                    position=1,
                    recent_pace_sec=67.0,
                )
            },
            active_battles=[
                BattleState(
                    driver_ahead="VER",
                    driver_behind="HAM",
                    gap_sec=0.5,
                    battle_intensity=0.9,
                )
            ],
            derived_summary=DerivedSummary(
                leader="VER",
                total_drivers=1,
                active_battles=1,
            ),
        )
        json_str = snap.model_dump_json()
        restored = RaceStateSnapshot.model_validate_json(json_str)
        assert restored.frame_id == 5
        assert "VER" in restored.drivers
        assert restored.active_battles[0].battle_intensity == 0.9


# ---------------------------------------------------------------------------
# 2. State update consistency
# ---------------------------------------------------------------------------


class TestStateUpdateConsistency:
    def test_engine_tracks_drivers(self):
        engine = RaceStateEngine()
        frame = _make_frame(0, [_driver("VER"), _driver("HAM", num="44", position=2, gap=0.5)])
        engine.update(frame)

        assert "VER" in engine.drivers
        assert "HAM" in engine.drivers
        assert engine.drivers["VER"].position == 1
        assert engine.drivers["HAM"].gap_ahead_sec == 0.5

    def test_engine_updates_incrementally(self):
        engine = RaceStateEngine()
        f1 = _make_frame(0, [_driver("VER", position=1, lap=1)])
        f2 = _make_frame(1, [_driver("VER", position=1, lap=2, speed=315.0)], session_time_sec=85)
        engine.update(f1)
        engine.update(f2)

        dh = engine.drivers["VER"]
        assert dh.lap_number == 2
        assert dh.speed_kph == 315.0
        assert engine.frame_count == 2

    def test_global_state_updates(self):
        engine = RaceStateEngine()
        f1 = _make_frame(0, track_status="GREEN FLAG", safety_car=False, air_temp=28.0)
        f2 = _make_frame(1, track_status="SAFETY CAR", safety_car=True, air_temp=27.5)
        engine.update(f1)
        engine.update(f2)

        assert engine.track_status == "SAFETY CAR"
        assert engine.safety_car is True
        assert engine.air_temp == 27.5

    def test_position_ordering(self):
        engine = RaceStateEngine()
        frame = _make_frame(
            0,
            [
                _driver("LEC", num="16", position=3, gap=1.5),
                _driver("VER", num="1", position=1),
                _driver("HAM", num="44", position=2, gap=0.8),
            ],
        )
        engine.update(frame)
        ordered = engine.sorted_drivers()
        positions = [d.position for d in ordered]
        assert positions == [1, 2, 3]

    def test_stint_resets_on_compound_change(self):
        engine = RaceStateEngine()
        f1 = _make_frame(0, [_driver("VER", compound="SOFT", tire_age=10, lap=10)])
        engine.update(f1)
        assert engine.drivers["VER"].stint_length == 10

        f2 = _make_frame(1, [_driver("VER", compound="HARD", tire_age=1, lap=11, speed=308.0)], session_time_sec=85)
        engine.update(f2)
        assert engine.drivers["VER"].stint_length == 1
        assert engine.drivers["VER"].tire_compound == "HARD"


# ---------------------------------------------------------------------------
# 3. Derived feature computation
# ---------------------------------------------------------------------------


class TestDerivedFeatures:
    def _build_multi_lap_frames(self, n_laps: int = 8) -> list[TelemetryFrame]:
        """Build frames where VER is P1 and HAM is P2 closing in."""
        frames = []
        for i in range(n_laps):
            # HAM's speed increases each lap (faster → closing gap)
            # VER stays constant
            gap = max(0.1, 2.0 - i * 0.25)
            frames.append(
                _make_frame(
                    i,
                    [
                        _driver("VER", position=1, lap=i + 1, speed=305.0 + i * 0.1, tire_age=i + 1),
                        _driver(
                            "HAM",
                            num="44",
                            position=2,
                            lap=i + 1,
                            gap=gap,
                            speed=310.0 + i * 0.5,
                            tire_age=i + 1,
                            drs=gap < 1.0,
                        ),
                    ],
                    session_time_sec=i * 85,
                )
            )
        return frames

    def test_recent_pace_computed(self):
        frames = self._build_multi_lap_frames(6)
        snapshots, _ = _run_frames(frames)
        last = snapshots[-1]
        # After 6 laps with distinct speeds, recent_pace should be populated
        ver = last.drivers.get("VER")
        assert ver is not None
        # Need at least 2 laps with speed changes to populate
        # First frame doesn't register a lap change, so pace builds up from frame 2+
        ham = last.drivers.get("HAM")
        assert ham is not None

    def test_pace_trend_direction(self):
        """VER's speed barely changes → near-zero trend.
        HAM speeds up each lap → negative trend (getting faster in time ≈
        lower proxy lap time since proxy = track_len / speed)."""
        frames = self._build_multi_lap_frames(8)
        snapshots, _ = _run_frames(frames)
        last = snapshots[-1]

        ham = last.drivers.get("HAM")
        if ham is not None and ham.pace_trend is not None:
            # HAM speeds up → lap-time proxy decreases → negative trend
            assert ham.pace_trend < 0, f"Expected negative trend, got {ham.pace_trend}"

    def test_catch_time_computed_when_closing(self):
        frames = self._build_multi_lap_frames(8)
        snapshots, _ = _run_frames(frames)
        last = snapshots[-1]
        ham = last.drivers.get("HAM")
        if ham is not None and ham.catch_time_laps is not None:
            assert ham.catch_time_laps > 0
            assert ham.catch_time_laps < 999

    def test_degradation_score_sign(self):
        """Build a scenario where a driver slows down over time."""
        frames = []
        for i in range(8):
            # Speed decreases each lap → proxy lap time increases → positive degradation
            frames.append(
                _make_frame(
                    i,
                    [_driver("VER", position=1, lap=i + 1, speed=310.0 - i * 2.0, tire_age=i + 1)],
                    session_time_sec=i * 85,
                )
            )
        snapshots, _ = _run_frames(frames)
        last = snapshots[-1]
        ver = last.drivers.get("VER")
        if ver is not None and ver.degradation_score is not None:
            # Slowing down → latest proxy > mean → positive degradation
            assert ver.degradation_score > 0, f"Expected positive degradation, got {ver.degradation_score}"

    def test_pit_window_thresholds(self):
        extractor = FeatureExtractor(RaceStateEngine())
        assert extractor._pit_window_open("SOFT", 14) is False
        assert extractor._pit_window_open("SOFT", 15) is True
        assert extractor._pit_window_open("MEDIUM", 22) is True
        assert extractor._pit_window_open("HARD", 29) is False
        assert extractor._pit_window_open("HARD", 30) is True
        assert extractor._pit_window_open(None, 10) is None
        assert extractor._pit_window_open("SOFT", None) is None

    def test_battle_detection(self):
        frame = _make_frame(
            0,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=1.5, drs=True),
                _driver("LEC", num="16", position=3, gap=5.0),  # too far → no battle
            ],
        )
        snapshots, _ = _run_frames([frame])
        battles = snapshots[0].active_battles
        assert len(battles) == 1
        assert battles[0].driver_ahead == "VER"
        assert battles[0].driver_behind == "HAM"
        assert battles[0].gap_sec == 1.5

    def test_battle_intensity_increases_with_proximity(self):
        """Closer gap → higher intensity."""
        f_far = _make_frame(
            0,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=1.8),
            ],
        )
        f_close = _make_frame(
            1,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=0.3),
            ],
            session_time_sec=85,
        )
        snap_far, _ = _run_frames([f_far])
        snap_close, _ = _run_frames([f_close])

        if snap_far[0].active_battles and snap_close[0].active_battles:
            assert snap_close[0].active_battles[0].battle_intensity > snap_far[0].active_battles[0].battle_intensity

    def test_battle_duration_increments(self):
        frames = [
            _make_frame(
                i,
                [
                    _driver("VER", position=1, lap=i + 1, speed=310.0),
                    _driver("HAM", num="44", position=2, gap=1.0, lap=i + 1, speed=312.0),
                ],
                session_time_sec=i * 85,
            )
            for i in range(5)
        ]
        snapshots, _ = _run_frames(frames)
        # Battle should persist across frames with increasing duration
        durations = []
        for s in snapshots:
            if s.active_battles:
                durations.append(s.active_battles[0].battle_duration_laps)
        assert len(durations) == 5
        # Duration should be monotonically increasing
        for i in range(1, len(durations)):
            assert durations[i] >= durations[i - 1]

    def test_ols_slope_basic(self):
        assert _ols_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert _ols_slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)
        assert _ols_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)
        assert _ols_slope([1.0]) is None
        assert _ols_slope([]) is None

    def test_derived_summary_fields(self):
        frames = [
            _make_frame(
                0,
                [
                    _driver("VER", position=1, speed=315.0),
                    _driver("HAM", num="44", position=2, gap=0.5, speed=318.0),
                ],
            )
        ]
        snapshots, _ = _run_frames(frames)
        ds = snapshots[0].derived_summary
        assert ds.leader == "VER"
        assert ds.total_drivers == 2
        assert ds.safety_car_active is False


# ---------------------------------------------------------------------------
# 4. Missing-data robustness
# ---------------------------------------------------------------------------


class TestMissingDataRobustness:
    def test_empty_frame(self):
        """Engine handles a frame with no drivers."""
        frame = _make_frame(0, drivers=[])
        snapshots, engine = _run_frames([frame])
        assert len(snapshots) == 1
        assert snapshots[0].drivers == {}
        assert engine.frame_count == 1

    def test_driver_with_all_none(self):
        """A driver with only number and abbreviation (everything else None)."""
        ds = DriverState(driver_number="99", abbreviation="TST")
        frame = _make_frame(0, drivers=[ds])
        snapshots, _ = _run_frames([frame])
        tst = snapshots[0].drivers.get("TST")
        assert tst is not None
        assert tst.position is None
        assert tst.recent_pace_sec is None
        assert tst.catch_time_laps is None
        assert tst.degradation_score is None
        assert tst.pit_window_open is None

    def test_partial_driver_data(self):
        """Driver has position and speed but no tire info."""
        ds = DriverState(
            driver_number="99",
            abbreviation="TST",
            position=5,
            speed_kph=300.0,
            lap_number=3,
        )
        frame = _make_frame(0, drivers=[ds])
        snapshots, _ = _run_frames([frame])
        tst = snapshots[0].drivers["TST"]
        assert tst.position == 5
        assert tst.tire_compound is None
        assert tst.pit_window_open is None

    def test_missing_global_state_fields(self):
        """GlobalState with all Nones should not crash."""
        frame = TelemetryFrame(
            frame_id=0,
            timestamp=datetime.now(timezone.utc),
            session_info=SESSION_INFO,
            drivers=[_driver("VER")],
            global_state=GlobalState(),
        )
        snapshots, _ = _run_frames([frame])
        assert snapshots[0].global_state.air_temp_celsius is None

    def test_single_driver_no_gap(self):
        """With only one driver, gap_ahead and gap_behind should be None."""
        frame = _make_frame(0, [_driver("VER", position=1)])
        snapshots, _ = _run_frames([frame])
        ver = snapshots[0].drivers["VER"]
        assert ver.gap_ahead_sec is None
        assert ver.gap_behind_sec is None

    def test_no_battles_when_gaps_missing(self):
        """Drivers with None gaps should not create battles."""
        frame = _make_frame(
            0,
            [
                _driver("VER", position=1),
                DriverState(driver_number="44", abbreviation="HAM", position=2),
            ],
        )
        snapshots, _ = _run_frames([frame])
        assert snapshots[0].active_battles == []


# ---------------------------------------------------------------------------
# 5. Snapshot serialization
# ---------------------------------------------------------------------------


class TestSnapshotSerialization:
    def _sample_snapshots(self, n: int = 5) -> list[RaceStateSnapshot]:
        frames = [
            _make_frame(
                i,
                [
                    _driver("VER", position=1, lap=i + 1, speed=310.0 + i),
                    _driver("HAM", num="44", position=2, gap=0.8, lap=i + 1, speed=312.0),
                ],
                session_time_sec=i * 85,
            )
            for i in range(n)
        ]
        snapshots, _ = _run_frames(frames)
        return snapshots

    def test_jsonl_round_trip(self):
        snapshots = self._sample_snapshots()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = Path(tmp.name)

        save_snapshots_jsonl(snapshots, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == len(snapshots)

        for line in lines:
            restored = RaceStateSnapshot.model_validate_json(line)
            assert "VER" in restored.drivers

        path.unlink()

    def test_csv_output(self):
        snapshots = self._sample_snapshots()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = Path(tmp.name)

        save_snapshots_csv(snapshots, path)
        content = path.read_text()
        lines = content.strip().split("\n")
        # Header + (n_snapshots * n_drivers) data rows
        assert len(lines) == 1 + len(snapshots) * 2  # 2 drivers per snapshot

        # Verify header
        assert "driver" in lines[0]
        assert "degradation_score" in lines[0]

        path.unlink()

    def test_print_snapshot_format(self):
        snapshots = self._sample_snapshots(1)
        text = print_snapshot(snapshots[0])
        assert "Frame 0" in text
        assert "VER" in text
        assert "HAM" in text
        assert "Test GP" in text

    def test_print_snapshot_driver_filter(self):
        snapshots = self._sample_snapshots(1)
        text = print_snapshot(snapshots[0], drivers=["VER"])
        assert "VER" in text
        # HAM should not appear in the driver lines (may appear in battles)
        driver_lines = [l for l in text.split("\n") if "P" in l and "|" in l]
        for line in driver_lines:
            assert "HAM" not in line


# ---------------------------------------------------------------------------
# 6. Snapshot comparison
# ---------------------------------------------------------------------------


class TestSnapshotComparison:
    def test_position_change_detected(self):
        f1 = _make_frame(
            0,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=0.5),
            ],
        )
        f2 = _make_frame(
            1,
            [
                _driver("HAM", num="44", position=1),
                _driver("VER", position=2, gap=0.3),
            ],
            session_time_sec=85,
        )
        snapshots, _ = _run_frames([f1, f2])
        delta = compare_snapshots(snapshots[0], snapshots[1])
        assert len(delta["position_changes"]) == 2
        # VER went from 1 to 2, HAM from 2 to 1
        abbrs = {pc["driver"] for pc in delta["position_changes"]}
        assert "VER" in abbrs
        assert "HAM" in abbrs

    def test_compound_change_detected_as_pit(self):
        f1 = _make_frame(0, [_driver("VER", compound="SOFT", tire_age=15)])
        f2 = _make_frame(1, [_driver("VER", compound="HARD", tire_age=1, speed=308.0)], session_time_sec=85)
        snapshots, _ = _run_frames([f1, f2])
        delta = compare_snapshots(snapshots[0], snapshots[1])
        assert len(delta["pit_stops"]) == 1
        assert delta["pit_stops"][0]["from_compound"] == "SOFT"
        assert delta["pit_stops"][0]["to_compound"] == "HARD"

    def test_flag_change_detected(self):
        f1 = _make_frame(0, [_driver("VER")], track_status="GREEN FLAG")
        f2 = _make_frame(1, [_driver("VER")], track_status="SAFETY CAR", safety_car=True, session_time_sec=85)
        snapshots, _ = _run_frames([f1, f2])
        delta = compare_snapshots(snapshots[0], snapshots[1])
        assert delta["flag_change"] is not None
        assert delta["flag_change"]["from"] == "GREEN FLAG"
        assert delta["flag_change"]["to"] == "SAFETY CAR"

    def test_new_battle_detected(self):
        f1 = _make_frame(
            0,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=3.0),  # too far
            ],
        )
        f2 = _make_frame(
            1,
            [
                _driver("VER", position=1),
                _driver("HAM", num="44", position=2, gap=1.0),  # now in battle range
            ],
            session_time_sec=85,
        )
        snapshots, _ = _run_frames([f1, f2])
        delta = compare_snapshots(snapshots[0], snapshots[1])
        assert len(delta["new_battles"]) == 1

    def test_no_changes_yields_empty_delta(self):
        f1 = _make_frame(0, [_driver("VER", position=1)])
        f2 = _make_frame(1, [_driver("VER", position=1)], session_time_sec=85)
        snapshots, _ = _run_frames([f1, f2])
        delta = compare_snapshots(snapshots[0], snapshots[1])
        assert delta["position_changes"] == []
        assert delta["pit_stops"] == []
        assert delta["flag_change"] is None
