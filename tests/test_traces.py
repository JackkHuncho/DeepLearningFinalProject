"""Tests for the Phase 12 trace logging, auditability, and reproducibility."""

import csv
import json
from pathlib import Path

import pytest

from f1_commentary.traces.trace_logger import TraceEntry, TraceLogger
from f1_commentary.traces.trace_utils import (
    export_traces_csv,
    export_traces_json,
    filter_traces,
    trace_summary,
)


# ---- 1. TraceEntry creates with defaults ----

def test_trace_entry_defaults():
    entry = TraceEntry()
    assert entry.trace_id  # non-empty UUID
    assert entry.beat_id == ""
    assert entry.telemetry_snapshot == {}
    assert entry.derived_features == {}
    assert entry.candidate_event == {}
    assert entry.scheduled_beat == {}
    assert entry.retrieval_context == {}
    assert entry.storyline_id is None
    assert entry.generated_commentary == {}
    assert entry.guard_decision == {}
    assert entry.final_commentary == {}
    assert entry.model_variant == ""
    assert entry.latency_breakdown == {}
    assert entry.total_latency_ms == 0.0


# ---- 2. TraceEntry serialization round-trip ----

def test_trace_entry_round_trip():
    entry = TraceEntry(
        beat_id="beat-1",
        storyline_id="story-1",
        model_variant="gpt-test",
        total_latency_ms=42.5,
        candidate_event={"candidate_type": "overtake_opportunity", "involved_drivers": ["HAM", "VER"]},
        final_commentary={"final_text": "Hamilton overtakes Verstappen!", "involved_drivers": ["HAM"]},
    )
    json_str = entry.model_dump_json()
    restored = TraceEntry.model_validate_json(json_str)
    assert restored.beat_id == entry.beat_id
    assert restored.storyline_id == entry.storyline_id
    assert restored.model_variant == entry.model_variant
    assert restored.total_latency_ms == entry.total_latency_ms
    assert restored.candidate_event == entry.candidate_event
    assert restored.final_commentary == entry.final_commentary
    assert restored.trace_id == entry.trace_id


# ---- 3. TraceLogger start_session creates file path ----

def test_trace_logger_start_session(tmp_path):
    logger = TraceLogger(tmp_path / "traces")
    path = logger.start_session("2024_Monza_R")
    assert path.parent.exists()
    assert "trace_2024_Monza_R_" in path.name
    assert path.suffix == ".jsonl"


# ---- 4. TraceLogger log + read_traces round-trip ----

def test_trace_logger_log_and_read(tmp_path):
    logger = TraceLogger(tmp_path / "traces")
    session_file = logger.start_session("test_session")

    entries = [
        TraceEntry(beat_id="beat-1", total_latency_ms=10.0),
        TraceEntry(beat_id="beat-2", total_latency_ms=20.0),
        TraceEntry(beat_id="beat-3", total_latency_ms=30.0),
    ]
    for e in entries:
        logger.log(e)

    read_back = logger.read_traces(session_file)
    assert len(read_back) == 3
    assert read_back[0].beat_id == "beat-1"
    assert read_back[1].beat_id == "beat-2"
    assert read_back[2].beat_id == "beat-3"
    assert read_back[0].trace_id == entries[0].trace_id


# ---- 5. TraceLogger log without session raises RuntimeError ----

def test_trace_logger_no_session_raises(tmp_path):
    logger = TraceLogger(tmp_path / "traces")
    with pytest.raises(RuntimeError, match="No active session"):
        logger.log(TraceEntry())


# ---- 6. filter_traces by driver ----

def test_filter_traces_by_driver():
    traces = [
        TraceEntry(candidate_event={"involved_drivers": ["HAM", "VER"]}),
        TraceEntry(candidate_event={"involved_drivers": ["LEC", "SAI"]}),
        TraceEntry(candidate_event={"involved_drivers": ["HAM", "NOR"]}),
    ]
    filtered = filter_traces(traces, driver="HAM")
    assert len(filtered) == 2
    filtered_lec = filter_traces(traces, driver="LEC")
    assert len(filtered_lec) == 1


# ---- 7. filter_traces by event_type ----

def test_filter_traces_by_event_type():
    traces = [
        TraceEntry(candidate_event={"candidate_type": "overtake_opportunity"}),
        TraceEntry(candidate_event={"candidate_type": "pit_stop"}),
        TraceEntry(candidate_event={"candidate_type": "overtake_opportunity"}),
    ]
    filtered = filter_traces(traces, event_type="overtake_opportunity")
    assert len(filtered) == 2
    filtered_pit = filter_traces(traces, event_type="pit_stop")
    assert len(filtered_pit) == 1


# ---- 8. filter_traces by storyline_id ----

def test_filter_traces_by_storyline_id():
    traces = [
        TraceEntry(storyline_id="battle-HAM-VER"),
        TraceEntry(storyline_id="pit-strategy"),
        TraceEntry(storyline_id="battle-HAM-VER"),
        TraceEntry(storyline_id=None),
    ]
    filtered = filter_traces(traces, storyline_id="battle-HAM-VER")
    assert len(filtered) == 2
    filtered_none = filter_traces(traces, storyline_id="pit-strategy")
    assert len(filtered_none) == 1


# ---- 9. trace_summary returns correct counts ----

def test_trace_summary_correct_counts():
    traces = [
        TraceEntry(
            candidate_event={"candidate_type": "overtake_opportunity", "involved_drivers": ["HAM", "VER"]},
            final_commentary={"involved_drivers": ["HAM"]},
            storyline_id="battle-1",
            total_latency_ms=100.0,
        ),
        TraceEntry(
            candidate_event={"candidate_type": "pit_stop", "involved_drivers": ["LEC"]},
            final_commentary={"involved_drivers": ["LEC"]},
            storyline_id="battle-1",
            total_latency_ms=200.0,
        ),
        TraceEntry(
            candidate_event={"candidate_type": "overtake_opportunity", "involved_drivers": ["NOR"]},
            final_commentary={"involved_drivers": []},
            storyline_id="battle-2",
            total_latency_ms=150.0,
        ),
    ]
    summary = trace_summary(traces)
    assert summary["total_traces"] == 3
    assert summary["event_type_distribution"]["overtake_opportunity"] == 2
    assert summary["event_type_distribution"]["pit_stop"] == 1
    assert summary["avg_latency_ms"] == 150.0
    assert summary["driver_mention_frequency"]["HAM"] == 2  # candidate + final
    assert summary["driver_mention_frequency"]["VER"] == 1
    assert summary["driver_mention_frequency"]["LEC"] == 2  # candidate + final
    assert summary["driver_mention_frequency"]["NOR"] == 1
    assert summary["storyline_counts"]["battle-1"] == 2
    assert summary["storyline_counts"]["battle-2"] == 1


# ---- 10. export_traces_csv produces valid CSV ----

def test_export_traces_csv(tmp_path):
    traces = [
        TraceEntry(
            beat_id="beat-1",
            candidate_event={"candidate_type": "overtake_opportunity", "involved_drivers": ["HAM", "VER"], "confidence": 0.9},
            final_commentary={"final_text": "Great overtake!", "confidence_band": "high"},
            total_latency_ms=42.0,
        ),
        TraceEntry(
            beat_id="beat-2",
            candidate_event={"candidate_type": "pit_stop", "involved_drivers": ["LEC"], "confidence": 0.7},
            final_commentary={"final_text": "Pit stop for Leclerc.", "confidence_band": "medium"},
            total_latency_ms=55.0,
        ),
    ]
    output_path = tmp_path / "traces.csv"
    export_traces_csv(traces, output_path)

    assert output_path.exists()
    with open(output_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["beat_id"] == "beat-1"
    assert rows[0]["candidate_type"] == "overtake_opportunity"
    assert rows[0]["involved_drivers"] == "HAM,VER"
    assert rows[1]["beat_id"] == "beat-2"
    assert rows[1]["final_text"] == "Pit stop for Leclerc."


# ---- 11. export_traces_json produces valid JSON ----

def test_export_traces_json(tmp_path):
    traces = [
        TraceEntry(
            beat_id="beat-1",
            storyline_id="story-1",
            total_latency_ms=42.0,
        ),
        TraceEntry(
            beat_id="beat-2",
            total_latency_ms=55.0,
        ),
    ]
    output_path = tmp_path / "traces.json"
    export_traces_json(traces, output_path)

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["beat_id"] == "beat-1"
    assert data[0]["storyline_id"] == "story-1"
    assert data[1]["beat_id"] == "beat-2"
    assert data[1]["total_latency_ms"] == 55.0
