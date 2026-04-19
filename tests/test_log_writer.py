"""Tests for JSONLLogWriter, ArchiveSinkConsumer, and StateRebuildConsumer."""

from datetime import datetime, timezone
from pathlib import Path

from f1_commentary.bus.archive_sink import ArchiveSinkConsumer
from f1_commentary.bus.log_writer import JSONLLogWriter
from f1_commentary.bus.state_rebuild import StateRebuildConsumer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    EventType,
    FinalCommentary,
    TelemetryFrame,
)

NOW = datetime.now(tz=timezone.utc)


def _telemetry(lap: int = 1) -> TelemetryFrame:
    return TelemetryFrame(timestamp=NOW, session_key="2024_Monza_R", lap=lap, drivers=[])


def _final(text: str = "Go!") -> FinalCommentary:
    return FinalCommentary(timestamp=NOW, final_text=text)


def _candidate() -> CandidateEvent:
    return CandidateEvent(
        timestamp=NOW,
        candidate_type=CandidateEventType.OVERTAKE_OPPORTUNITY,
        involved_drivers=["HAM", "VER"],
        confidence=0.8,
    )


# ---- JSONLLogWriter -------------------------------------------------------

class TestJSONLLogWriter:
    def test_write_and_read_all(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry(1))
        writer.write(_telemetry(2))
        events = writer.read_all()
        assert len(events) == 2
        assert events[0]["lap"] == 1
        assert events[1]["lap"] == 2

    def test_append_only(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry(1))
        # Create a second writer to the same file — should append
        writer2 = JSONLLogWriter(log)
        writer2.write(_telemetry(2))
        assert len(writer.read_all()) == 2

    def test_read_all_empty(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.jsonl"
        writer = JSONLLogWriter(log)
        assert writer.read_all() == []

    def test_read_by_type(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry())
        writer.write(_final())
        writer.write(_telemetry())

        tel = writer.read_by_type(EventType.TELEMETRY_FRAME)
        assert len(tel) == 2
        fin = writer.read_by_type(EventType.FINAL_COMMENTARY)
        assert len(fin) == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = tmp_path / "deep" / "nested" / "dir" / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry())
        assert log.exists()


# ---- ArchiveSinkConsumer --------------------------------------------------

class TestArchiveSinkConsumer:
    def test_routes_by_type(self, tmp_path: Path) -> None:
        archive = ArchiveSinkConsumer(tmp_path / "archive")
        archive.handle_event(_telemetry())
        archive.handle_event(_final())
        archive.handle_event(_telemetry())

        tel_file = tmp_path / "archive" / "telemetry_frame.jsonl"
        fin_file = tmp_path / "archive" / "final_commentary.jsonl"
        assert tel_file.exists()
        assert fin_file.exists()
        assert len(tel_file.read_text().strip().splitlines()) == 2
        assert len(fin_file.read_text().strip().splitlines()) == 1


# ---- StateRebuildConsumer -------------------------------------------------

class TestStateRebuildConsumer:
    def test_rebuild_basic(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry(5))
        writer.write(_telemetry(10))
        writer.write(_final("Hamilton wins"))

        consumer = StateRebuildConsumer()
        state = consumer.rebuild(log)
        assert state["success"] is True
        assert state["total_events"] == 3
        assert state["last_telemetry_lap"] == 10
        assert state["last_session_key"] == "2024_Monza_R"
        assert state["final_commentaries"] == ["Hamilton wins"]

    def test_rebuild_nonexistent(self, tmp_path: Path) -> None:
        consumer = StateRebuildConsumer()
        state = consumer.rebuild(tmp_path / "nope.jsonl")
        assert state["success"] is False
        assert state["total_events"] == 0

    def test_rebuild_counts(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        writer.write(_telemetry())
        writer.write(_candidate())
        writer.write(_candidate())
        writer.write(_final())

        state = StateRebuildConsumer().rebuild(log)
        assert state["event_counts"]["telemetry_frame"] == 1
        assert state["event_counts"]["candidate_event"] == 2
        assert state["event_counts"]["final_commentary"] == 1

    def test_write_and_rebuild_consistency(self, tmp_path: Path) -> None:
        """Write N events, rebuild, verify counts match."""
        log = tmp_path / "events.jsonl"
        writer = JSONLLogWriter(log)
        n_tel, n_fin = 5, 3
        for i in range(n_tel):
            writer.write(_telemetry(i))
        for _ in range(n_fin):
            writer.write(_final("text"))

        all_events = writer.read_all()
        assert len(all_events) == n_tel + n_fin

        state = StateRebuildConsumer().rebuild(log)
        assert state["total_events"] == n_tel + n_fin
        assert state["event_counts"]["telemetry_frame"] == n_tel
        assert state["event_counts"]["final_commentary"] == n_fin
        assert len(state["final_commentaries"]) == n_fin
