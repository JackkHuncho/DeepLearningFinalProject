"""Comprehensive tests for Phase 7: SFT dataset creation pipeline.

Coverage
========
- Transcript ingestion (JSON, JSONL, CSV, plain text)
- Timestamp parsing and normalisation
- EventWindow extraction
- Transcript alignment (strict and loose modes)
- SFT formatter (fact/inference extraction, input text formatting)
- Quality checks and deduplication
- Dataset splitting (deterministic, reproducible)
- Schema validation
- Edge cases
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.race_state import (
    BattleState,
    DerivedSummary,
    DriverRaceState,
    GlobalRaceStateSummary,
    RaceStateSnapshot,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScoreBreakdown,
    ScheduledBeat,
    SuppressionReason,
)
from src.telemetry_to_narrative.schemas.sft_dataset import (
    AlignmentConfidence,
    DatasetMetadata,
    EventWindow,
    SFTExample,
    SourceRef,
    TranscriptSegment,
)
from src.telemetry_to_narrative.training.transcript_loader import (
    load_csv,
    load_json,
    load_jsonl,
    load_plain_text,
    load_transcript,
    parse_timestamp,
)
from src.telemetry_to_narrative.training.event_window_builder import (
    extract_event_windows,
    _build_state_summary,
)
from src.telemetry_to_narrative.training.transcript_aligner import (
    AlignmentResult,
    align_transcripts,
    _heuristic_match_score,
)
from src.telemetry_to_narrative.training.sft_formatter import (
    extract_observed_facts,
    extract_derived_inferences,
    format_input_text,
    format_sft_example,
    format_sft_dataset,
)
from src.telemetry_to_narrative.training.quality_checks import (
    CONFIDENCE_RANK,
    check_duplicate_targets,
    check_empty_targets,
    check_long_targets,
    check_short_targets,
    check_unmatched,
    confidence_at_least,
    deduplicate,
    filter_by_confidence,
    run_quality_checks,
)
from src.telemetry_to_narrative.training.dataset_builder import split_dataset


# ── Fixtures ─────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
SESSION_INFO = SessionInfo(year=2024, grand_prix="Monza", session_type="R")


def _make_snapshot(frame_id: int, ts_offset_sec: int = 0, **kwargs) -> RaceStateSnapshot:
    """Create a minimal RaceStateSnapshot for testing."""
    ts = BASE_TS + timedelta(seconds=ts_offset_sec)
    drivers = kwargs.get("drivers", {
        "VER": DriverRaceState(
            driver_number="1", abbreviation="VER", position=1,
            lap_number=5, gap_ahead_sec=0.0, tire_compound="MEDIUM",
            tire_age_laps=5, speed_kph=310.0,
            pace_trend=-0.05, degradation_score=0.2, pit_window_open=False,
        ),
        "HAM": DriverRaceState(
            driver_number="44", abbreviation="HAM", position=2,
            lap_number=5, gap_ahead_sec=0.8, tire_compound="MEDIUM",
            tire_age_laps=5, speed_kph=308.0,
            pace_trend=0.15, degradation_score=0.6, pit_window_open=True,
        ),
    })
    battles = kwargs.get("active_battles", [
        BattleState(
            driver_ahead="VER", driver_behind="HAM",
            gap_sec=0.8, pace_delta_sec=-0.1, battle_intensity=0.7,
            battle_duration_laps=3,
        ),
    ])
    return RaceStateSnapshot(
        frame_id=frame_id,
        timestamp=ts,
        session_info=SESSION_INFO,
        global_state=GlobalRaceStateSummary(track_status="GREEN FLAG"),
        drivers=drivers,
        active_battles=battles,
        derived_summary=DerivedSummary(
            leader="VER", total_drivers=2, active_battles=1,
        ),
    )


def _make_beat(frame_id: int, selected: bool = True) -> ScheduledBeat:
    """Create a minimal ScheduledBeat for testing."""
    ts = BASE_TS + timedelta(seconds=frame_id * 10)
    return ScheduledBeat(
        beat_id=f"beat_{frame_id}",
        timestamp=ts,
        event_type=EventType.LEAD_BATTLE,
        involved_drivers=["VER", "HAM"],
        storyline_id="battle_HAM_VER",
        priority_score=0.75,
        score_breakdown=ScoreBreakdown(),
        selected_for_generation=selected,
        suppression_reason=SuppressionReason.NONE if selected else SuppressionReason.LOW_PRIORITY,
        source_frame_id=frame_id,
        description="VER vs HAM lead battle",
    )


def _make_segment(ts_offset_sec: int, text: str, **kwargs) -> TranscriptSegment:
    ts = BASE_TS + timedelta(seconds=ts_offset_sec)
    return TranscriptSegment(
        segment_id=f"seg_{ts_offset_sec}",
        timestamp=ts,
        text=text,
        **kwargs,
    )


# =====================================================================
# Transcript ingestion tests
# =====================================================================


class TestTimestampParsing:
    """Tests for parse_timestamp."""

    def test_iso_8601(self):
        dt = parse_timestamp("2024-03-15T14:30:00")
        assert dt.year == 2024
        assert dt.hour == 14
        assert dt.tzinfo == timezone.utc

    def test_iso_8601_with_tz(self):
        dt = parse_timestamp("2024-03-15T14:30:00+0000")
        assert dt.hour == 14

    def test_iso_space_separator(self):
        dt = parse_timestamp("2024-03-15 14:30:00")
        assert dt.year == 2024
        assert dt.hour == 14

    def test_hms_only(self):
        ref = datetime(2024, 6, 1, tzinfo=timezone.utc)
        dt = parse_timestamp("14:30:00", reference_date=ref)
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.hour == 14
        assert dt.minute == 30

    def test_hms_default_reference(self):
        dt = parse_timestamp("08:15:30")
        assert dt.year == 2024  # default reference
        assert dt.hour == 8

    def test_epoch_seconds(self):
        dt = parse_timestamp("1704067200")
        assert dt.year == 2024

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_timestamp("not-a-timestamp")


class TestLoadJSON:
    """Tests for JSON transcript loading."""

    def test_load_bare_list(self, tmp_path):
        data = [
            {"timestamp": "2024-01-01T14:00:00", "text": "Hello world"},
            {"timestamp": "2024-01-01T14:01:00", "text": "Second segment"},
        ]
        p = tmp_path / "transcript.json"
        p.write_text(json.dumps(data))
        segments = load_json(p)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"

    def test_load_segments_key(self, tmp_path):
        data = {"segments": [
            {"timestamp": "2024-01-01T14:00:00", "text": "Test"},
        ]}
        p = tmp_path / "transcript.json"
        p.write_text(json.dumps(data))
        segments = load_json(p)
        assert len(segments) == 1

    def test_drivers_mentioned_string(self, tmp_path):
        data = [{"timestamp": "2024-01-01T14:00:00", "text": "VER leads",
                 "drivers_mentioned": "VER, HAM"}]
        p = tmp_path / "transcript.json"
        p.write_text(json.dumps(data))
        segments = load_json(p)
        assert segments[0].drivers_mentioned == ["VER", "HAM"]

    def test_all_fields(self, tmp_path):
        data = [{
            "timestamp": "2024-01-01T14:00:00",
            "end_timestamp": "2024-01-01T14:00:10",
            "text": "Commentary here",
            "speaker": "Crofty",
            "drivers_mentioned": ["VER"],
            "event_type_hint": "lead_battle",
            "segment_id": "custom_id",
        }]
        p = tmp_path / "transcript.json"
        p.write_text(json.dumps(data))
        segments = load_json(p)
        seg = segments[0]
        assert seg.segment_id == "custom_id"
        assert seg.speaker == "Crofty"
        assert seg.event_type_hint == "lead_battle"
        assert seg.end_timestamp is not None


class TestLoadJSONL:
    """Tests for JSONL transcript loading."""

    def test_basic(self, tmp_path):
        lines = [
            json.dumps({"timestamp": "2024-01-01T14:00:00", "text": "Line 1"}),
            json.dumps({"timestamp": "2024-01-01T14:01:00", "text": "Line 2"}),
        ]
        p = tmp_path / "transcript.jsonl"
        p.write_text("\n".join(lines))
        segments = load_jsonl(p)
        assert len(segments) == 2

    def test_blank_lines_skipped(self, tmp_path):
        lines = [
            json.dumps({"timestamp": "2024-01-01T14:00:00", "text": "Line 1"}),
            "",
            json.dumps({"timestamp": "2024-01-01T14:01:00", "text": "Line 2"}),
        ]
        p = tmp_path / "transcript.jsonl"
        p.write_text("\n".join(lines))
        segments = load_jsonl(p)
        assert len(segments) == 2


class TestLoadCSV:
    """Tests for CSV transcript loading."""

    def test_basic(self, tmp_path):
        csv_text = "timestamp,text\n2024-01-01T14:00:00,Hello world\n2024-01-01T14:01:00,Second\n"
        p = tmp_path / "transcript.csv"
        p.write_text(csv_text)
        segments = load_csv(p)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"

    def test_optional_columns(self, tmp_path):
        csv_text = "timestamp,text,speaker,drivers_mentioned\n2024-01-01T14:00:00,Test,Brundle,\"VER, HAM\"\n"
        p = tmp_path / "transcript.csv"
        p.write_text(csv_text)
        segments = load_csv(p)
        assert segments[0].speaker == "Brundle"
        assert segments[0].drivers_mentioned == ["VER", "HAM"]


class TestLoadPlainText:
    """Tests for plain-text transcript loading."""

    def test_bracket_format(self, tmp_path):
        txt = "[14:00:00] Verstappen leads from Hamilton\n[14:01:00] Norris overtakes Leclerc\n"
        p = tmp_path / "transcript.txt"
        p.write_text(txt)
        segments = load_plain_text(p)
        assert len(segments) == 2
        assert "Verstappen" in segments[0].text

    def test_dash_format(self, tmp_path):
        txt = "14:00:00 - First line\n14:01:00 - Second line\n"
        p = tmp_path / "transcript.txt"
        p.write_text(txt)
        segments = load_plain_text(p)
        assert len(segments) == 2
        assert segments[0].text == "First line"

    def test_unrecognised_lines_skipped(self, tmp_path):
        txt = "[14:00:00] Good line\nBad line without timestamp\n[14:01:00] Another good line\n"
        p = tmp_path / "transcript.txt"
        p.write_text(txt)
        segments = load_plain_text(p)
        assert len(segments) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        segments = load_plain_text(p)
        assert len(segments) == 0


class TestLoadTranscript:
    """Tests for the unified auto-detect loader."""

    def test_auto_detect_json(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps([{"timestamp": "2024-01-01T14:00:00", "text": "Test"}]))
        segments = load_transcript(p)
        assert len(segments) == 1

    def test_auto_detect_txt(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("[14:00:00] Test line\n")
        segments = load_transcript(p)
        assert len(segments) == 1

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "t.xml"
        p.write_text("<xml/>")
        with pytest.raises(ValueError, match="Unsupported"):
            load_transcript(p)

    def test_results_sorted_by_timestamp(self, tmp_path):
        data = [
            {"timestamp": "2024-01-01T14:02:00", "text": "Later"},
            {"timestamp": "2024-01-01T14:00:00", "text": "Earlier"},
        ]
        p = tmp_path / "t.json"
        p.write_text(json.dumps(data))
        segments = load_transcript(p)
        assert segments[0].text == "Earlier"
        assert segments[1].text == "Later"


# =====================================================================
# EventWindow extraction tests
# =====================================================================


class TestEventWindowBuilder:
    """Tests for event window extraction."""

    def test_basic_extraction(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(5)]
        beats = [_make_beat(2, selected=True)]
        windows = extract_event_windows(snaps, beats, context_radius=1)
        assert len(windows) == 1
        w = windows[0]
        assert w.centre_frame_id == 2
        assert len(w.frame_ids) == 3  # [1, 2, 3]
        assert w.event_type == EventType.LEAD_BATTLE

    def test_selected_only_filter(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(5)]
        beats = [_make_beat(1, selected=True), _make_beat(3, selected=False)]
        windows = extract_event_windows(snaps, beats, selected_only=True)
        assert len(windows) == 1

    def test_include_suppressed(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(5)]
        beats = [_make_beat(1, selected=True), _make_beat(3, selected=False)]
        windows = extract_event_windows(snaps, beats, selected_only=False)
        assert len(windows) == 2

    def test_edge_clamping_start(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(5)]
        beats = [_make_beat(0, selected=True)]
        windows = extract_event_windows(snaps, beats, context_radius=2)
        assert len(windows) == 1
        assert windows[0].frame_ids[0] == 0  # clamped to start

    def test_edge_clamping_end(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(5)]
        beats = [_make_beat(4, selected=True)]
        windows = extract_event_windows(snaps, beats, context_radius=2)
        assert len(windows) == 1
        assert windows[0].frame_ids[-1] == 4  # clamped to end

    def test_empty_snapshots(self):
        windows = extract_event_windows([], [_make_beat(0)])
        assert len(windows) == 0

    def test_missing_frame_id_skipped(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(3)]
        beats = [_make_beat(99, selected=True)]  # frame 99 doesn't exist
        windows = extract_event_windows(snaps, beats)
        assert len(windows) == 0

    def test_state_summary_content(self):
        snaps = [_make_snapshot(i, ts_offset_sec=i * 10) for i in range(3)]
        beats = [_make_beat(1, selected=True)]
        windows = extract_event_windows(snaps, beats)
        summary = windows[0].state_summary
        assert summary["leader"] == "VER"
        assert "VER" in summary["drivers"]
        assert "HAM" in summary["drivers"]
        assert len(summary["battles"]) >= 1


# =====================================================================
# Transcript alignment tests
# =====================================================================


class TestTranscriptAlignment:
    """Tests for transcript alignment logic."""

    def _make_window(self, ts_offsets: list[int], drivers: list[str] = None) -> EventWindow:
        """Helper to create an EventWindow with given timestamp offsets."""
        timestamps = [BASE_TS + timedelta(seconds=o) for o in ts_offsets]
        return EventWindow(
            window_id=f"win_{ts_offsets[0]}",
            frame_ids=list(range(len(ts_offsets))),
            timestamps=timestamps,
            centre_frame_id=ts_offsets[len(ts_offsets) // 2],
            centre_timestamp=timestamps[len(timestamps) // 2],
            session_info=SESSION_INFO,
            event_type=EventType.LEAD_BATTLE,
            involved_drivers=drivers or ["VER", "HAM"],
        )

    def test_exact_match(self):
        window = self._make_window([0, 10, 20])
        segment = _make_segment(10, "Commentary at exact time", drivers_mentioned=["VER"])
        results = align_transcripts([window], [segment], mode="strict")
        assert results[0].confidence == AlignmentConfidence.EXACT
        assert results[0].segment == segment

    def test_near_match_loose(self):
        window = self._make_window([0, 10, 20])
        segment = _make_segment(25, "Commentary slightly after window")
        results = align_transcripts([window], [segment], mode="loose", tolerance_sec=10)
        assert results[0].confidence == AlignmentConfidence.NEAR

    def test_near_match_rejected_strict(self):
        window = self._make_window([0, 10, 20])
        segment = _make_segment(25, "Commentary slightly after window")
        results = align_transcripts([window], [segment], mode="strict")
        assert results[0].confidence == AlignmentConfidence.UNMATCHED

    def test_heuristic_match(self):
        window = self._make_window([0, 10, 20], drivers=["VER", "HAM"])
        segment = _make_segment(
            500, "Verstappen and Hamilton battle",
            drivers_mentioned=["VER", "HAM"],
            event_type_hint="lead_battle",
        )
        results = align_transcripts([window], [segment], mode="loose")
        assert results[0].confidence == AlignmentConfidence.HEURISTIC

    def test_unmatched_window(self):
        window = self._make_window([0, 10, 20], drivers=["VER"])
        # Segment with no matching drivers, no timestamp match.
        segment = _make_segment(500, "Unrelated commentary")
        results = align_transcripts([window], [segment], mode="strict")
        assert results[0].confidence == AlignmentConfidence.UNMATCHED
        assert results[0].segment is None

    def test_no_double_assignment(self):
        """Each segment should only be assigned to one window."""
        w1 = self._make_window([0, 10, 20])
        w2 = self._make_window([10, 20, 30])
        segment = _make_segment(15, "Single segment")
        results = align_transcripts([w1, w2], [segment], mode="loose")
        assigned = [r for r in results if r.segment is not None]
        assert len(assigned) == 1  # only one window gets the segment

    def test_empty_segments(self):
        window = self._make_window([0, 10, 20])
        results = align_transcripts([window], [], mode="loose")
        assert len(results) == 1
        assert results[0].confidence == AlignmentConfidence.UNMATCHED

    def test_empty_windows(self):
        segment = _make_segment(10, "Segment")
        results = align_transcripts([], [segment], mode="loose")
        assert len(results) == 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            align_transcripts([], [], mode="invalid")


class TestHeuristicScore:
    """Tests for the heuristic matching function."""

    def test_full_driver_overlap(self):
        segment = _make_segment(0, "text", drivers_mentioned=["VER", "HAM"])
        window = EventWindow(
            window_id="w", frame_ids=[0], timestamps=[BASE_TS],
            centre_frame_id=0, centre_timestamp=BASE_TS,
            session_info=SESSION_INFO, event_type=EventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
        )
        score = _heuristic_match_score(segment, window)
        assert score >= 0.4

    def test_event_type_match(self):
        segment = _make_segment(0, "text", event_type_hint="lead_battle")
        window = EventWindow(
            window_id="w", frame_ids=[0], timestamps=[BASE_TS],
            centre_frame_id=0, centre_timestamp=BASE_TS,
            session_info=SESSION_INFO, event_type=EventType.LEAD_BATTLE,
            involved_drivers=[],
        )
        score = _heuristic_match_score(segment, window)
        assert score >= 0.5

    def test_no_overlap(self):
        segment = _make_segment(0, "text")
        window = EventWindow(
            window_id="w", frame_ids=[0], timestamps=[BASE_TS],
            centre_frame_id=0, centre_timestamp=BASE_TS,
            session_info=SESSION_INFO, event_type=EventType.LEAD_BATTLE,
            involved_drivers=["NOR"],
        )
        score = _heuristic_match_score(segment, window)
        assert score == 0.0


# =====================================================================
# SFT formatter tests
# =====================================================================


class TestSFTFormatter:
    """Tests for SFT example formatting."""

    def _make_window_with_summary(self) -> EventWindow:
        snap = _make_snapshot(1, ts_offset_sec=10)
        summary = _build_state_summary(snap, ["VER", "HAM"])
        return EventWindow(
            window_id="w1",
            frame_ids=[0, 1, 2],
            timestamps=[BASE_TS, BASE_TS + timedelta(seconds=10), BASE_TS + timedelta(seconds=20)],
            centre_frame_id=1,
            centre_timestamp=BASE_TS + timedelta(seconds=10),
            session_info=SESSION_INFO,
            event_type=EventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
            state_summary=summary,
        )

    def test_extract_observed_facts(self):
        window = self._make_window_with_summary()
        facts = extract_observed_facts(window)
        assert any("VER" in f for f in facts)
        assert any("lap" in f.lower() for f in facts)

    def test_extract_derived_inferences(self):
        window = self._make_window_with_summary()
        inferences = extract_derived_inferences(window)
        # HAM has degradation_score=0.6 and pit_window_open=True
        assert any("degradation" in inf.lower() for inf in inferences)
        assert any("pit window" in inf.lower() for inf in inferences)

    def test_format_input_text_structure(self):
        text = format_input_text(
            ["Fact 1", "Fact 2"],
            ["Inference 1"],
            [],
            EventType.LEAD_BATTLE,
            ["VER", "HAM"],
        )
        assert "[Event: lead_battle" in text
        assert "## Observed Facts" in text
        assert "- Fact 1" in text
        assert "## Derived Inferences" in text
        assert "- Inference 1" in text
        assert "## Context" in text
        assert "(none available)" in text

    def test_format_input_text_no_facts(self):
        text = format_input_text([], [], [], EventType.OVERTAKE, ["NOR"])
        assert "(no facts available)" in text
        assert "(no inferences available)" in text

    def test_format_sft_example_matched(self):
        window = self._make_window_with_summary()
        segment = _make_segment(10, "Great racing between Verstappen and Hamilton!")
        alignment = AlignmentResult(window, segment, AlignmentConfidence.EXACT)
        example = format_sft_example(alignment)

        assert example.event_type == EventType.LEAD_BATTLE
        assert len(example.observed_facts) > 0
        assert example.target_text == "Great racing between Verstappen and Hamilton!"
        assert example.alignment_confidence == AlignmentConfidence.EXACT
        assert len(example.source_refs) == 1
        assert example.input_text  # non-empty

    def test_format_sft_example_unmatched(self):
        window = self._make_window_with_summary()
        alignment = AlignmentResult(window, None, AlignmentConfidence.UNMATCHED)
        example = format_sft_example(alignment)
        assert example.target_text == ""
        assert example.alignment_confidence == AlignmentConfidence.UNMATCHED

    def test_format_sft_dataset_batch(self):
        window = self._make_window_with_summary()
        seg = _make_segment(10, "Commentary text")
        alignments = [
            AlignmentResult(window, seg, AlignmentConfidence.EXACT),
            AlignmentResult(window, None, AlignmentConfidence.UNMATCHED),
        ]
        examples = format_sft_dataset(alignments)
        assert len(examples) == 2


# =====================================================================
# Quality checks tests
# =====================================================================


class TestQualityChecks:
    """Tests for data quality checks and deduplication."""

    def _make_example(self, target: str, confidence: AlignmentConfidence = AlignmentConfidence.EXACT, **kwargs) -> SFTExample:
        return SFTExample(
            example_id=kwargs.get("example_id", f"ex_{id(target)}"),
            session_info=SESSION_INFO,
            event_type=EventType.LEAD_BATTLE,
            input_text="[Event: lead_battle]\n## Observed Facts\n- test",
            target_text=target,
            alignment_confidence=confidence,
        )

    def test_empty_targets(self):
        examples = [
            self._make_example("Good text", example_id="e1"),
            self._make_example("", example_id="e2"),
            self._make_example("   ", example_id="e3"),
        ]
        empty = check_empty_targets(examples)
        assert "e2" in empty
        assert "e3" in empty
        assert "e1" not in empty

    def test_short_targets(self):
        examples = [
            self._make_example("One two three four five", example_id="e1"),
            self._make_example("Short", example_id="e2"),
        ]
        short = check_short_targets(examples, min_words=5)
        assert "e2" in short
        assert "e1" not in short

    def test_long_targets(self):
        long_text = " ".join(["word"] * 501)
        examples = [
            self._make_example("Normal length text here today", example_id="e1"),
            self._make_example(long_text, example_id="e2"),
        ]
        long = check_long_targets(examples, max_words=500)
        assert "e2" in long
        assert "e1" not in long

    def test_duplicate_targets(self):
        examples = [
            self._make_example("Duplicate text here", example_id="e1"),
            self._make_example("Unique text here", example_id="e2"),
            self._make_example("Duplicate text here", example_id="e3"),
        ]
        dups = check_duplicate_targets(examples)
        assert "e1" in dups
        assert "e3" in dups
        assert "e2" not in dups

    def test_unmatched_check(self):
        examples = [
            self._make_example("Text", confidence=AlignmentConfidence.EXACT, example_id="e1"),
            self._make_example("Text", confidence=AlignmentConfidence.UNMATCHED, example_id="e2"),
        ]
        unmatched = check_unmatched(examples)
        assert "e2" in unmatched
        assert "e1" not in unmatched

    def test_deduplicate_keeps_first(self):
        examples = [
            self._make_example("Same text", example_id="e1"),
            self._make_example("Same text", example_id="e2"),
            self._make_example("Different text", example_id="e3"),
        ]
        deduped = deduplicate(examples)
        assert len(deduped) == 2
        ids = [e.example_id for e in deduped]
        assert "e1" in ids
        assert "e3" in ids

    def test_run_quality_checks_filters(self):
        examples = [
            self._make_example("Good commentary text here and more", example_id="e1"),
            self._make_example("", example_id="e2"),  # empty
            self._make_example("Also good text for training here", confidence=AlignmentConfidence.UNMATCHED, example_id="e3"),
        ]
        filtered, report = run_quality_checks(examples, min_confidence=AlignmentConfidence.NEAR)
        assert len(filtered) == 1  # only e1 survives (e2 empty, e3 below NEAR)
        assert report["total_before_filtering"] == 3
        assert report["removed"] == 2

    def test_run_quality_checks_no_filter(self):
        examples = [
            self._make_example("Good text here for the model", example_id="e1"),
        ]
        filtered, report = run_quality_checks(examples)
        assert len(filtered) == 1
        assert report["removed"] == 0


# =====================================================================
# Confidence filtering tests
# =====================================================================


class TestConfidenceFiltering:
    """Tests for confidence-based filtering."""

    def _make_example(self, confidence: AlignmentConfidence, example_id: str) -> SFTExample:
        return SFTExample(
            example_id=example_id,
            session_info=SESSION_INFO,
            event_type=EventType.LEAD_BATTLE,
            input_text="[Event: lead_battle]\n## Observed Facts\n- test",
            target_text=f"Commentary for {example_id}",
            alignment_confidence=confidence,
        )

    def _make_mixed_examples(self) -> list[SFTExample]:
        """Create one example per confidence level."""
        return [
            self._make_example(AlignmentConfidence.EXACT, "ex_exact"),
            self._make_example(AlignmentConfidence.NEAR, "ex_near"),
            self._make_example(AlignmentConfidence.HEURISTIC, "ex_heuristic"),
            self._make_example(AlignmentConfidence.UNMATCHED, "ex_unmatched"),
        ]

    # -- confidence_at_least --

    def test_confidence_ordering(self):
        assert CONFIDENCE_RANK[AlignmentConfidence.EXACT] > CONFIDENCE_RANK[AlignmentConfidence.NEAR]
        assert CONFIDENCE_RANK[AlignmentConfidence.NEAR] > CONFIDENCE_RANK[AlignmentConfidence.HEURISTIC]
        assert CONFIDENCE_RANK[AlignmentConfidence.HEURISTIC] > CONFIDENCE_RANK[AlignmentConfidence.UNMATCHED]

    def test_exact_meets_exact(self):
        assert confidence_at_least(AlignmentConfidence.EXACT, AlignmentConfidence.EXACT)

    def test_near_meets_near(self):
        assert confidence_at_least(AlignmentConfidence.NEAR, AlignmentConfidence.NEAR)

    def test_exact_meets_heuristic(self):
        assert confidence_at_least(AlignmentConfidence.EXACT, AlignmentConfidence.HEURISTIC)

    def test_heuristic_does_not_meet_near(self):
        assert not confidence_at_least(AlignmentConfidence.HEURISTIC, AlignmentConfidence.NEAR)

    def test_unmatched_does_not_meet_heuristic(self):
        assert not confidence_at_least(AlignmentConfidence.UNMATCHED, AlignmentConfidence.HEURISTIC)

    def test_unmatched_meets_unmatched(self):
        assert confidence_at_least(AlignmentConfidence.UNMATCHED, AlignmentConfidence.UNMATCHED)

    # -- filter_by_confidence --

    def test_filter_exact_only(self):
        examples = self._make_mixed_examples()
        kept, dropped = filter_by_confidence(examples, AlignmentConfidence.EXACT)
        assert len(kept) == 1
        assert kept[0].example_id == "ex_exact"
        assert dropped == 3

    def test_filter_near_threshold(self):
        examples = self._make_mixed_examples()
        kept, dropped = filter_by_confidence(examples, AlignmentConfidence.NEAR)
        assert len(kept) == 2
        ids = {e.example_id for e in kept}
        assert ids == {"ex_exact", "ex_near"}
        assert dropped == 2

    def test_filter_heuristic_threshold(self):
        examples = self._make_mixed_examples()
        kept, dropped = filter_by_confidence(examples, AlignmentConfidence.HEURISTIC)
        assert len(kept) == 3
        ids = {e.example_id for e in kept}
        assert ids == {"ex_exact", "ex_near", "ex_heuristic"}
        assert dropped == 1

    def test_filter_unmatched_keeps_all(self):
        examples = self._make_mixed_examples()
        kept, dropped = filter_by_confidence(examples, AlignmentConfidence.UNMATCHED)
        assert len(kept) == 4
        assert dropped == 0

    def test_filter_preserves_alignment_field(self):
        """Filtering should never alter the stored alignment_confidence."""
        examples = self._make_mixed_examples()
        kept, _ = filter_by_confidence(examples, AlignmentConfidence.HEURISTIC)
        for ex in kept:
            assert ex.alignment_confidence in (
                AlignmentConfidence.EXACT, AlignmentConfidence.NEAR, AlignmentConfidence.HEURISTIC,
            )
        # Specifically check the heuristic one is still labeled HEURISTIC.
        heur = [e for e in kept if e.example_id == "ex_heuristic"]
        assert len(heur) == 1
        assert heur[0].alignment_confidence == AlignmentConfidence.HEURISTIC

    def test_filter_deterministic(self):
        """Same input + same threshold → same output."""
        examples = self._make_mixed_examples()
        kept1, _ = filter_by_confidence(examples, AlignmentConfidence.NEAR)
        kept2, _ = filter_by_confidence(examples, AlignmentConfidence.NEAR)
        assert [e.example_id for e in kept1] == [e.example_id for e in kept2]

    # -- run_quality_checks with min_confidence --

    def test_quality_checks_exact_threshold(self):
        examples = self._make_mixed_examples()
        filtered, report = run_quality_checks(
            examples, min_confidence=AlignmentConfidence.EXACT,
            remove_empty=False, remove_duplicates=False,
        )
        assert len(filtered) == 1
        assert report["confidence_filter_dropped"] == 3
        assert report["summary"]["min_confidence_threshold"] == "exact"

    def test_quality_checks_heuristic_threshold(self):
        examples = self._make_mixed_examples()
        filtered, report = run_quality_checks(
            examples, min_confidence=AlignmentConfidence.HEURISTIC,
            remove_empty=False, remove_duplicates=False,
        )
        assert len(filtered) == 3
        assert report["confidence_filter_dropped"] == 1

    def test_quality_checks_default_near(self):
        """Default min_confidence is NEAR."""
        examples = self._make_mixed_examples()
        filtered, report = run_quality_checks(
            examples, remove_empty=False, remove_duplicates=False,
        )
        assert len(filtered) == 2
        assert report["summary"]["min_confidence_threshold"] == "near"

    def test_quality_checks_counts_before_and_after(self):
        examples = self._make_mixed_examples()
        _, report = run_quality_checks(
            examples, min_confidence=AlignmentConfidence.NEAR,
            remove_empty=False, remove_duplicates=False,
        )
        before = report["summary"]["confidence_counts_before"]
        after = report["summary"]["confidence_counts_after"]
        assert before["exact"] == 1
        assert before["near"] == 1
        assert before["heuristic"] == 1
        assert before["unmatched"] == 1
        assert after["exact"] == 1
        assert after["near"] == 1
        assert after["heuristic"] == 0
        assert after["unmatched"] == 0


# =====================================================================
# Dataset split tests
# =====================================================================


class TestDatasetSplit:
    """Tests for deterministic dataset splitting."""

    def _make_examples(self, n: int) -> list[SFTExample]:
        return [
            SFTExample(
                example_id=f"ex_{i}",
                session_info=SESSION_INFO,
                event_type=EventType.LEAD_BATTLE,
                input_text=f"input {i}",
                target_text=f"target {i}",
            )
            for i in range(n)
        ]

    def test_split_sizes(self):
        examples = self._make_examples(10)
        train, val = split_dataset(examples, val_fraction=0.2, seed=42)
        assert len(train) + len(val) == 10
        assert len(val) == 2
        assert len(train) == 8

    def test_split_deterministic(self):
        examples = self._make_examples(20)
        train1, val1 = split_dataset(examples, seed=42)
        train2, val2 = split_dataset(examples, seed=42)
        assert [e.example_id for e in train1] == [e.example_id for e in train2]
        assert [e.example_id for e in val1] == [e.example_id for e in val2]

    def test_different_seed_different_split(self):
        examples = self._make_examples(20)
        train1, _ = split_dataset(examples, seed=42)
        train2, _ = split_dataset(examples, seed=99)
        ids1 = [e.example_id for e in train1]
        ids2 = [e.example_id for e in train2]
        assert ids1 != ids2

    def test_small_dataset(self):
        examples = self._make_examples(2)
        train, val = split_dataset(examples, val_fraction=0.5, seed=42)
        assert len(train) + len(val) == 2
        assert len(train) >= 1

    def test_single_example(self):
        examples = self._make_examples(1)
        train, val = split_dataset(examples, val_fraction=0.2, seed=42)
        assert len(train) == 1
        assert len(val) == 0


# =====================================================================
# Schema validation tests
# =====================================================================


class TestSchemaValidation:
    """Tests for Pydantic schema serialisation / deserialisation."""

    def test_sft_example_roundtrip(self):
        example = SFTExample(
            example_id="test_123",
            session_info=SESSION_INFO,
            event_type=EventType.OVERTAKE,
            involved_drivers=["NOR", "LEC"],
            observed_facts=["NOR P3 on lap 10"],
            derived_inferences=["pace improving"],
            input_text="[Event: overtake]\n## Observed Facts\n- NOR P3",
            target_text="Norris overtakes Leclerc!",
            alignment_confidence=AlignmentConfidence.EXACT,
        )
        json_str = example.model_dump_json()
        restored = SFTExample.model_validate_json(json_str)
        assert restored.example_id == "test_123"
        assert restored.event_type == EventType.OVERTAKE
        assert restored.alignment_confidence == AlignmentConfidence.EXACT

    def test_event_window_roundtrip(self):
        window = EventWindow(
            window_id="w1",
            frame_ids=[1, 2, 3],
            timestamps=[BASE_TS, BASE_TS + timedelta(seconds=10), BASE_TS + timedelta(seconds=20)],
            centre_frame_id=2,
            centre_timestamp=BASE_TS + timedelta(seconds=10),
            session_info=SESSION_INFO,
            event_type=EventType.PIT_STRATEGY,
            involved_drivers=["SAI"],
            state_summary={"lap": 15},
        )
        json_str = window.model_dump_json()
        restored = EventWindow.model_validate_json(json_str)
        assert restored.window_id == "w1"
        assert len(restored.frame_ids) == 3

    def test_dataset_metadata_roundtrip(self):
        meta = DatasetMetadata(
            created_at=BASE_TS,
            total_examples=100,
            train_examples=80,
            val_examples=20,
            event_type_distribution={"lead_battle": 30, "overtake": 20},
            alignment_distribution={"exact": 50, "near": 30},
        )
        json_str = meta.model_dump_json()
        restored = DatasetMetadata.model_validate_json(json_str)
        assert restored.total_examples == 100

    def test_transcript_segment_roundtrip(self):
        seg = TranscriptSegment(
            segment_id="s1",
            timestamp=BASE_TS,
            end_timestamp=BASE_TS + timedelta(seconds=5),
            speaker="Crofty",
            text="Lights out!",
            drivers_mentioned=["VER", "HAM"],
            event_type_hint="race_control",
        )
        json_str = seg.model_dump_json()
        restored = TranscriptSegment.model_validate_json(json_str)
        assert restored.speaker == "Crofty"
        assert len(restored.drivers_mentioned) == 2

    def test_source_ref_defaults(self):
        ref = SourceRef(snapshot_frame_id=5)
        assert ref.beat_id is None
        assert ref.transcript_segment_id is None


# =====================================================================
# Sample transcript file test
# =====================================================================


class TestSampleTranscript:
    """Verify the sample transcript file can be loaded."""

    def test_load_sample_commentary(self):
        p = Path(__file__).parent.parent / "data" / "transcripts" / "sample_commentary.json"
        if not p.exists():
            pytest.skip("Sample transcript file not found")
        segments = load_transcript(p)
        assert len(segments) == 10
        assert all(seg.text for seg in segments)
        # Check sorted order.
        for i in range(len(segments) - 1):
            assert segments[i].timestamp <= segments[i + 1].timestamp
