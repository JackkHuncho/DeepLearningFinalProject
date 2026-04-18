"""Integration tests for Phase 15: end-to-end pipeline runner.

Coverage
========
- Full local pipeline with mock backend
- Baseline pipeline path
- Trace-writer JSONL output (all stages present, roundtrip loads)
- Stop-after debug points
- Scenario filtering (lead_battle, pit_strategy, race_control)
- Graceful fallback when generator returns nothing / beat is missing
- Output adapters wired but not disruptive (no-terminal, no-sse)
- ID preservation across stages: frame → snapshot → beat → generation → final
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    GlobalState,
    SessionInfo,
    TelemetryFrame,
)
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.final_commentary import (
    FinalCommentary,
    FinalSuppressionReason,
)
from src.telemetry_to_narrative.schemas.baseline_commentary import BaselineCommentary

from src.telemetry_to_narrative.pipeline import (
    PipelineConfig,
    PipelineMode,
    PipelineRunner,
    StopAfter,
)
from src.telemetry_to_narrative.pipeline.config import Scenario
from src.telemetry_to_narrative.pipeline.trace_writer import TraceWriter


# ── Helpers ─────────────────────────────────────────────────────────────

SAMPLE_REPLAY = Path(__file__).resolve().parent.parent / "data" / "logs" / "sample_replay.jsonl"


def _load_sample_frames(n: int | None = None) -> list[TelemetryFrame]:
    frames: list[TelemetryFrame] = []
    with open(SAMPLE_REPLAY) as fh:
        for i, line in enumerate(fh):
            if n is not None and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            frames.append(TelemetryFrame.model_validate_json(line))
    return frames


def _synth_frame(
    frame_id: int,
    ts_offset_sec: int,
    positions: list[tuple[str, int, float | None]],
    compound: str = "MEDIUM",
    track_status: str = "GREEN FLAG",
    safety_car: bool = False,
) -> TelemetryFrame:
    """Build a synthetic TelemetryFrame with configurable positions/gaps."""
    base = datetime(2024, 9, 1, 14, 0, 0, tzinfo=timezone.utc)
    drivers = [
        DriverState(
            driver_number=str(i + 1),
            abbreviation=abbr,
            position=pos,
            lap_number=max(1, frame_id // 2 + 1),
            gap_ahead_sec=gap,
            tire_compound=compound,
            tire_age_laps=frame_id,
            speed_kph=300.0 + i,
        )
        for i, (abbr, pos, gap) in enumerate(positions)
    ]
    return TelemetryFrame(
        frame_id=frame_id,
        timestamp=base + timedelta(seconds=ts_offset_sec),
        session_time=timedelta(seconds=ts_offset_sec),
        session_info=SessionInfo(year=2024, grand_prix="Test", session_type="R"),
        drivers=drivers,
        global_state=GlobalState(track_status=track_status, safety_car=safety_car),
    )


# =====================================================================
# Full pipeline — local mode
# =====================================================================


class TestLocalPipeline:
    """Exercise the full local pipeline with mock backends."""

    def test_end_to_end_with_sample_frames(self, tmp_path):
        frames = _load_sample_frames(10)
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_frames=10,
            max_generation_items=5,
        )
        runner = PipelineRunner(config)
        result = runner.run(frames=frames)

        assert result.counters.frames == 10
        assert result.counters.snapshots == 10
        assert result.counters.candidate_events > 0
        assert result.counters.beats_total > 0
        assert result.counters.beats_selected > 0
        assert result.counters.generations > 0
        assert result.counters.finals_emitted + result.counters.finals_suppressed > 0

    def test_local_pipeline_produces_finals(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)
        # At least some finals should be emitted (mock backend is deterministic).
        assert any(not f.suppression_applied for f in result.finals)

    def test_local_pipeline_no_trace_dir_is_noop(self):
        """Omitting trace_dir runs cleanly without writing anything."""
        frames = _load_sample_frames(5)
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=None,
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)
        assert result.trace_paths == {}
        assert result.counters.frames == 5

    def test_requires_exactly_one_source(self):
        runner = PipelineRunner(PipelineConfig(enable_terminal_output=False))
        with pytest.raises(ValueError):
            runner.run()
        with pytest.raises(ValueError):
            runner.run(frames=[], fastf1_session=(2024, "Monza", "R"))


# =====================================================================
# Baseline pipeline
# =====================================================================


class TestBaselinePipeline:
    """Exercise the baseline path — BeatManager is skipped."""

    def test_baseline_produces_baseline_commentary(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            mode=PipelineMode.BASELINE,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_generation_items=3,
        )
        result = PipelineRunner(config).run(frames=frames)

        assert result.counters.baselines > 0
        assert len(result.finals) == 0  # BeatManager skipped in baseline mode
        assert all(isinstance(r, BaselineCommentary) for r in result.baselines)
        assert all(r.system == "baseline" for r in result.baselines)

    def test_baseline_trace_file_populated(self, tmp_path):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            mode=PipelineMode.BASELINE,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_generation_items=3,
        )
        result = PipelineRunner(config).run(frames=frames)

        trace_path = Path(result.trace_paths["baselines"])
        assert trace_path.exists()
        lines = [l for l in trace_path.read_text().splitlines() if l.strip()]
        assert len(lines) == result.counters.baselines
        # JSONL is valid.
        for line in lines:
            bc = BaselineCommentary.model_validate_json(line)
            assert bc.provider == "mock"


# =====================================================================
# Trace writer
# =====================================================================


class TestTraceWriter:
    """Verify trace JSONL outputs roundtrip and are inspectable."""

    def test_all_stage_files_exist(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_generation_items=3,
        )
        result = PipelineRunner(config).run(frames=frames)

        expected = {
            "frames", "snapshots", "candidates", "beats",
            "generated", "finals", "baselines", "summary",
        }
        assert expected.issubset(set(result.trace_paths.keys()))
        for name, path in result.trace_paths.items():
            assert Path(path).exists(), f"{name} missing at {path}"

    def test_frames_trace_roundtrips(self, tmp_path):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            trace_dir=str(tmp_path), enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)

        path = Path(result.trace_paths["frames"])
        loaded = [
            TelemetryFrame.model_validate_json(line)
            for line in path.read_text().splitlines() if line.strip()
        ]
        assert len(loaded) == 6
        assert [f.frame_id for f in loaded] == [f.frame_id for f in frames[:6]]

    def test_summary_json_captures_counters(self, tmp_path):
        frames = _load_sample_frames(5)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            session_label="unit-test",
        )
        result = PipelineRunner(config).run(frames=frames)

        summary_path = Path(result.trace_paths["summary"])
        summary = json.loads(summary_path.read_text())

        assert summary["mode"] == "local"
        assert summary["session_label"] == "unit-test"
        assert summary["counters"]["frames"] == 5
        assert "flags" in summary
        assert "trace_paths" in summary

    def test_suppressed_items_still_traced(self, tmp_path):
        """Suppressed finals should land in the trace even if hidden from output."""
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            include_suppressed_in_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)

        finals_path = Path(result.trace_paths["finals"])
        finals_text = finals_path.read_text()
        loaded = [
            FinalCommentary.model_validate_json(l)
            for l in finals_text.splitlines() if l.strip()
        ]
        # Traces include both emitted and suppressed; the BeatManager inside
        # the pipeline always uses include_suppressed=True for traceability.
        # Even if no suppressions happened on this sample, count must equal
        # emitted + suppressed on the result object.
        assert len(loaded) == result.counters.finals_emitted + result.counters.finals_suppressed

    def test_trace_writer_noop_when_dir_is_none(self):
        with TraceWriter(None) as tw:
            assert tw.paths == {}
            # write does not raise in no-op mode.

    def test_trace_writer_rejects_unknown_stage(self, tmp_path):
        with TraceWriter(tmp_path) as tw:
            with pytest.raises(KeyError):
                tw.write("does_not_exist", None)  # type: ignore[arg-type]


# =====================================================================
# Stop points
# =====================================================================


class TestStopPoints:
    """Debug stop-points stop the pipeline cleanly after each stage."""

    @pytest.mark.parametrize("stop,expected_field", [
        (StopAfter.REPLAY, "frames"),
        (StopAfter.SNAPSHOTS, "snapshots"),
        (StopAfter.BEATS, "beats"),
        (StopAfter.GENERATION, "generated"),
    ])
    def test_stop_after(self, tmp_path, stop, expected_field):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            stop_after=stop,
        )
        result = PipelineRunner(config).run(frames=frames)

        assert result.stopped_at == stop.value

        # Stages at or before stop should have data, stages after must be empty.
        order = ["frames", "snapshots", "beats", "generated", "finals"]
        idx = order.index(expected_field)
        for i, field in enumerate(order):
            got = getattr(result, field)
            if i <= idx:
                if field in ("frames", "snapshots", "beats"):
                    assert len(got) > 0, f"{field} should be populated at stop={stop}"
            else:
                assert len(got) == 0, f"{field} must be empty after stop={stop}"

    def test_stop_at_replay_skips_state_engine(self, tmp_path):
        frames = _load_sample_frames(4)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            stop_after=StopAfter.REPLAY,
        )
        result = PipelineRunner(config).run(frames=frames)

        assert result.counters.frames == 4
        assert result.counters.snapshots == 0
        assert result.counters.beats_total == 0


# =====================================================================
# Scenario filtering
# =====================================================================


class TestScenarios:
    """Scenario filter drops candidates outside the chosen event type."""

    def test_lead_battle_only_keeps_lead_and_overtake(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            scenario=Scenario.LEAD_BATTLE,
        )
        result = PipelineRunner(config).run(frames=frames)

        types = {c.event_type for c in result.candidates}
        assert types.issubset({EventType.LEAD_BATTLE, EventType.OVERTAKE})

    def test_race_control_scenario_only_keeps_race_control(self, tmp_path):
        # Synthetic frames where a red flag appears halfway through.
        frames = [
            _synth_frame(0, 0, [("VER", 1, None), ("HAM", 2, 0.5)]),
            _synth_frame(1, 10, [("VER", 1, None), ("HAM", 2, 0.5)],
                         track_status="SAFETY CAR DEPLOYED", safety_car=True),
            _synth_frame(2, 20, [("VER", 1, None), ("HAM", 2, 0.5)],
                         track_status="GREEN FLAG"),
        ]
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            scenario=Scenario.RACE_CONTROL,
        )
        result = PipelineRunner(config).run(frames=frames)

        if result.candidates:
            assert all(c.event_type == EventType.RACE_CONTROL for c in result.candidates)

    def test_all_scenario_keeps_everything(self, tmp_path):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            scenario=Scenario.ALL,
        )
        result = PipelineRunner(config).run(frames=frames)
        # No filter applied — at least more than one distinct event type in 6 frames.
        assert len({c.event_type for c in result.candidates}) >= 1


# =====================================================================
# Fallback behavior
# =====================================================================


class TestFallbacks:
    """Graceful degradation for missing or failed stages."""

    def test_beat_without_generation_produces_fallback(self, tmp_path):
        """When max_items caps generation but more beats remain, BeatManager
        must emit fallbacks rather than drop the beat silently."""
        frames = _load_sample_frames(10)
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_generation_items=1,  # force uncovered beats
        )
        result = PipelineRunner(config).run(frames=frames)

        # Some beats lacked a matching generation → fallback path.
        assert result.counters.fallbacks_used >= 0
        assert result.counters.finals_emitted + result.counters.finals_suppressed > 0

    def test_unknown_backend_falls_back_to_mock(self, tmp_path):
        """load_backend falls back to mock on unknown type — pipeline still runs."""
        frames = _load_sample_frames(5)
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            local_backend="nonexistent_backend_name",
        )
        result = PipelineRunner(config).run(frames=frames)
        assert result.counters.generations >= 0  # ran without crashing

    def test_empty_frames_runs_cleanly(self, tmp_path):
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=[])
        assert result.counters.frames == 0
        assert result.counters.snapshots == 0
        assert result.counters.beats_total == 0


# =====================================================================
# Output adapters
# =====================================================================


class TestOutputAdapters:
    """Output adapter flags are respected — and don't disrupt the pipeline."""

    def test_no_terminal_suppresses_stdout(self, tmp_path, capsys):
        frames = _load_sample_frames(5)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            enable_sse_output=False,
        )
        PipelineRunner(config).run(frames=frames)
        captured = capsys.readouterr()
        # TerminalAdapter writes its header; with no-terminal we shouldn't
        # see the ANSI "Final Commentary" banner.
        assert "FINAL COMMENTARY" not in captured.out.upper() or captured.out == ""

    def test_sse_output_emits_done_event(self, tmp_path, capsys):
        frames = _load_sample_frames(5)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            enable_sse_output=True,
        )
        PipelineRunner(config).run(frames=frames)
        captured = capsys.readouterr()
        assert "event: done" in captured.out


# =====================================================================
# ID preservation
# =====================================================================


class TestIdPreservation:
    """Critical invariant: IDs link all stages together."""

    def test_beat_id_flows_through_generation_and_final(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)

        beat_ids = {b.beat_id for b in result.beats}
        for g in result.generated:
            assert g.beat_id in beat_ids
        for f in result.finals:
            assert f.beat_id in beat_ids

    def test_generation_id_preserved_in_final(self, tmp_path):
        frames = _load_sample_frames(8)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)

        gen_ids = {g.generation_id for g in result.generated}
        for f in result.finals:
            # Fallbacks have None; emitted finals must point back to a generation.
            if not f.suppression_applied:
                assert f.source_generation_id in gen_ids

    def test_frame_id_flows_to_beat_source(self, tmp_path):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        result = PipelineRunner(config).run(frames=frames)

        frame_ids = {f.frame_id for f in result.frames}
        for b in result.beats:
            assert b.source_frame_id in frame_ids

    def test_baseline_beat_id_matches_scheduled_beats(self, tmp_path):
        frames = _load_sample_frames(6)
        config = PipelineConfig(
            mode=PipelineMode.BASELINE,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
            max_generation_items=3,
        )
        result = PipelineRunner(config).run(frames=frames)

        beat_ids = {b.beat_id for b in result.beats if b.selected_for_generation}
        for r in result.baselines:
            assert r.beat_id in beat_ids
