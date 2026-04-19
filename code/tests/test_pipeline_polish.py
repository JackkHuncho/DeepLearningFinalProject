"""Phase 17 polish tests — config validation, CLI help, doctor, trace
linkage, and JSONL schema compatibility.

These tests do not exercise generation logic (covered in Phase 15) —
they pin down the new robustness/usability surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.main import cli
from src.telemetry_to_narrative.pipeline.config import (
    PipelineConfig,
    PipelineMode,
    Scenario,
    StopAfter,
)
from src.telemetry_to_narrative.pipeline.diagnostics import (
    check_jsonl_schema,
    check_trace_schemas,
    load_run_summary,
    run_doctor,
    validate_trace_linkage,
)
from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.pipeline import PipelineRunner


REPO_CODE = Path(__file__).resolve().parent.parent       # .../code
SAMPLE_REPLAY = REPO_CODE / "data" / "logs" / "sample_replay.jsonl"
SAMPLE_LOCAL = REPO_CODE / "data" / "artifacts" / "sample_run_local"


def _load_sample_frames(n: int = 10) -> list[TelemetryFrame]:
    frames: list[TelemetryFrame] = []
    with open(SAMPLE_REPLAY) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            line = line.strip()
            if line:
                frames.append(TelemetryFrame.model_validate_json(line))
    return frames


# ── Config defaults & validation ─────────────────────────────────────


class TestPipelineConfigDefaults:
    def test_defaults_are_safe(self):
        c = PipelineConfig()
        assert c.mode == PipelineMode.LOCAL
        assert c.stop_after == StopAfter.NONE
        assert c.scenario == Scenario.ALL
        assert c.local_backend == "mock"
        assert c.baseline_backend == "mock"
        assert c.enable_beat_manager is True
        assert c.realtime is False
        assert 0.0 <= c.temperature <= 2.0
        # Default config validates cleanly.
        c.validate()

    @pytest.mark.parametrize("kwargs,bad_field", [
        ({"max_frames": -1}, "max_frames"),
        ({"max_generation_items": -3}, "max_generation_items"),
        ({"replay_speed": 0}, "replay_speed"),
        ({"replay_speed": -2.0}, "replay_speed"),
        ({"beat_refresh_interval": -1}, "beat_refresh_interval"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": -8}, "max_new_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.5}, "temperature"),
    ])
    def test_validate_rejects_out_of_range(self, kwargs, bad_field):
        c = PipelineConfig(**kwargs)
        with pytest.raises(ValueError) as exc:
            c.validate()
        assert bad_field in str(exc.value)

    def test_validate_collects_multiple_errors(self):
        c = PipelineConfig(max_frames=-1, temperature=5.0, max_new_tokens=0)
        with pytest.raises(ValueError) as exc:
            c.validate()
        msg = str(exc.value)
        assert "max_frames" in msg
        assert "temperature" in msg
        assert "max_new_tokens" in msg

    def test_unknown_backend_does_not_block_validate(self):
        # Backends fall back to mock at runtime — validate must not reject.
        PipelineConfig(local_backend="totally-fake").validate()
        PipelineConfig(baseline_backend="also-fake").validate()

    def test_runner_calls_validate(self):
        # Fail-fast: bad config should error before any frames are read.
        bad = PipelineConfig(temperature=99.0, enable_terminal_output=False)
        with pytest.raises(ValueError):
            PipelineRunner(bad).run(frames=[])


# ── CLI help / discoverability ───────────────────────────────────────


class TestCLIHelp:
    def setup_method(self):
        self.runner = CliRunner()

    def test_root_help_lists_pipeline_commands(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("run-pipeline", "run-baseline-pipeline",
                    "run-scenario", "doctor", "validate-trace"):
            assert cmd in result.output

    @pytest.mark.parametrize("cmd", [
        "run-pipeline", "run-baseline-pipeline", "run-scenario",
        "doctor", "validate-trace",
    ])
    def test_per_command_help_renders(self, cmd):
        result = self.runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_pipeline_help_mentions_stop_after_choices(self):
        result = self.runner.invoke(cli, ["run-pipeline", "--help"])
        assert "replay" in result.output and "snapshots" in result.output
        assert "beats" in result.output and "generation" in result.output

    def test_pipeline_help_includes_example(self):
        result = self.runner.invoke(cli, ["run-pipeline", "--help"])
        assert "Example" in result.output or "example" in result.output


# ── Doctor ───────────────────────────────────────────────────────────


class TestDoctor:
    def test_doctor_returns_structured_report(self):
        report = run_doctor(REPO_CODE)
        assert "folders" in report
        assert "dependencies" in report
        assert "backends" in report
        assert "samples" in report
        # Mock backends are always reachable.
        assert report["backends"]["local:mock"] == "ready"
        assert report["backends"]["baseline:mock"] == "ready"

    def test_doctor_detects_missing_folders(self, tmp_path):
        report = run_doctor(tmp_path)
        assert report["ok"] is False
        assert report["folders"]["data"]["exists"] is False

    def test_doctor_cli_exits_zero_when_healthy(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--repo-root", str(REPO_CODE)])
        assert result.exit_code == 0
        assert "doctor" in result.output

    def test_doctor_cli_json_mode(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["doctor", "--repo-root", str(REPO_CODE), "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "backends" in payload
        assert payload["backends"]["local:mock"] == "ready"


# ── Trace linkage + schema compat ────────────────────────────────────


class TestTraceLinkage:
    def test_sample_local_run_links_cleanly(self):
        report = validate_trace_linkage(SAMPLE_LOCAL)
        assert report["ok"] is True, report["errors"]
        assert report["counts"]["frames"] > 0
        assert report["counts"]["beats"] > 0

    def test_sample_local_schemas_validate(self):
        report = check_trace_schemas(SAMPLE_LOCAL)
        assert report["ok"] is True
        # Every present file fully validated.
        for fname, info in report["files"].items():
            if "skipped" in info:
                continue
            assert info["valid"] == info["total"], (fname, info)

    def test_summary_loads(self):
        summary = load_run_summary(SAMPLE_LOCAL)
        assert summary is not None
        assert "counters" in summary and "trace_paths" in summary

    def test_linkage_after_fresh_run(self, tmp_path):
        # End-to-end: run pipeline, then validate its own traces.
        from src.telemetry_to_narrative.pipeline import (
            PipelineConfig, PipelineMode, PipelineRunner,
        )
        config = PipelineConfig(
            mode=PipelineMode.LOCAL,
            trace_dir=str(tmp_path),
            enable_terminal_output=False,
        )
        PipelineRunner(config).run(frames=_load_sample_frames(8))

        linkage = validate_trace_linkage(tmp_path)
        assert linkage["ok"] is True, linkage["errors"]
        schemas = check_trace_schemas(tmp_path)
        assert schemas["ok"] is True

    def test_linkage_detects_orphan_beats(self, tmp_path):
        # Inject a beat that points at a nonexistent frame.
        from src.telemetry_to_narrative.pipeline import (
            PipelineConfig, PipelineRunner,
        )
        PipelineRunner(
            PipelineConfig(trace_dir=str(tmp_path), enable_terminal_output=False)
        ).run(frames=_load_sample_frames(5))

        beats_path = tmp_path / "scheduled_beats.jsonl"
        with open(beats_path) as fh:
            lines = fh.readlines()
        assert lines, "expected at least one beat from sample run"

        # Re-emit with one beat re-pointed to a frame_id that does not exist.
        bad = json.loads(lines[0])
        bad["source_frame_id"] = 999_999
        lines.append(json.dumps(bad) + "\n")
        beats_path.write_text("".join(lines))

        linkage = validate_trace_linkage(tmp_path)
        assert linkage["ok"] is False
        assert any("source_frame_id" in e for e in linkage["errors"])


class TestSchemaCompat:
    def test_check_jsonl_schema_passes_on_clean_file(self):
        report = check_jsonl_schema(
            SAMPLE_LOCAL / "replay_frames.jsonl", TelemetryFrame,
        )
        assert report["total"] == report["valid"]
        assert report["errors"] == []

    def test_check_jsonl_schema_flags_bad_lines(self, tmp_path):
        bad_path = tmp_path / "broken.jsonl"
        bad_path.write_text(
            '{"frame_id": "not-an-int"}\n'
            '{"this_is_not_a_telemetry_frame": true}\n'
            'not even json\n'
        )
        report = check_jsonl_schema(bad_path, TelemetryFrame)
        assert report["total"] == 3
        assert report["valid"] == 0
        assert len(report["errors"]) == 3

    def test_check_jsonl_schema_handles_missing_file(self, tmp_path):
        report = check_jsonl_schema(
            tmp_path / "does-not-exist.jsonl", TelemetryFrame,
        )
        assert report["errors"]
        assert report["total"] == 0


# ── Scenario / stop_after string parsing through CLI ─────────────────


class TestCLIArgPlumbing:
    def test_invalid_scenario_choice_rejected(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-pipeline", "--from-jsonl", str(SAMPLE_REPLAY),
             "--scenario", "not_a_scenario", "--max-frames", "1"],
        )
        assert result.exit_code != 0

    def test_invalid_stop_after_rejected(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-pipeline", "--from-jsonl", str(SAMPLE_REPLAY),
             "--stop-after", "elsewhere", "--max-frames", "1"],
        )
        assert result.exit_code != 0
