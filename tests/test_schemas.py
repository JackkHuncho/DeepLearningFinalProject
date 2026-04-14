"""Tests for event schema definitions."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from f1_commentary.schemas.events import (
    BaseEvent,
    CandidateEvent,
    CandidateEventType,
    ClaimSupport,
    DriverState,
    EventType,
    FinalCommentary,
    GeneratedCommentary,
    GuardedCommentary,
    RaceStateSnapshot,
    RetrievalResult,
    ScheduledBeat,
    TelemetryFrame,
)


NOW = datetime.now(tz=timezone.utc)


def _driver(code: str = "HAM", pos: int = 1, lap: int = 10) -> DriverState:
    return DriverState(driver_id=code, position=pos, lap_number=lap)


# ---- DriverState ---------------------------------------------------------

class TestDriverState:
    def test_minimal(self) -> None:
        d = DriverState(driver_id="VER", position=1, lap_number=5)
        assert d.driver_id == "VER"
        assert d.gap_ahead is None

    def test_full(self) -> None:
        d = DriverState(
            driver_id="HAM",
            position=2,
            lap_number=10,
            gap_ahead=1.2,
            tire_compound="SOFT",
            tire_age=8,
            drs_available=True,
            speed=320.5,
            last_lap_time=78.123,
        )
        assert d.tire_compound == "SOFT"

    def test_roundtrip(self) -> None:
        d = DriverState(driver_id="LEC", position=3, lap_number=7, gap_ahead=0.5)
        d2 = DriverState.model_validate_json(d.model_dump_json())
        assert d == d2


# ---- TelemetryFrame ------------------------------------------------------

class TestTelemetryFrame:
    def test_create(self) -> None:
        tf = TelemetryFrame(
            timestamp=NOW,
            session_key="2024_Monza_R",
            lap=12,
            drivers=[_driver()],
        )
        assert tf.event_type == EventType.TELEMETRY_FRAME
        assert tf.event_id  # auto-generated UUID

    def test_roundtrip(self) -> None:
        tf = TelemetryFrame(
            timestamp=NOW,
            session_key="2024_Monza_R",
            lap=12,
            drivers=[_driver(), _driver("VER", 2, 12)],
            flag_status="GREEN",
            weather="dry",
        )
        tf2 = TelemetryFrame.model_validate_json(tf.model_dump_json())
        assert tf2.session_key == tf.session_key
        assert len(tf2.drivers) == 2


# ---- RaceStateSnapshot ---------------------------------------------------

class TestRaceStateSnapshot:
    def test_create_with_derived(self) -> None:
        rs = RaceStateSnapshot(
            timestamp=NOW,
            session_key="2024_Monza_R",
            lap=20,
            drivers=[_driver()],
            gaps={"HAM": 0.0, "VER": 1.5},
            battle_pairs=[("HAM", "VER")],
        )
        assert rs.event_type == EventType.RACE_STATE_UPDATE
        assert rs.battle_pairs == [("HAM", "VER")]

    def test_roundtrip(self) -> None:
        rs = RaceStateSnapshot(
            timestamp=NOW,
            session_key="s",
            lap=1,
            drivers=[_driver()],
            pace_deltas={"HAM": -0.3},
        )
        rs2 = RaceStateSnapshot.model_validate_json(rs.model_dump_json())
        assert rs2.pace_deltas == rs.pace_deltas


# ---- CandidateEvent ------------------------------------------------------

class TestCandidateEvent:
    def test_create(self) -> None:
        ce = CandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.OVERTAKE_OPPORTUNITY,
            involved_drivers=["HAM", "VER"],
            confidence=0.85,
        )
        assert ce.event_type == EventType.CANDIDATE_EVENT
        assert ce.severity == 0.5  # default

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CandidateEvent(
                timestamp=NOW,
                candidate_type=CandidateEventType.PIT_STOP,
                involved_drivers=["HAM"],
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            CandidateEvent(
                timestamp=NOW,
                candidate_type=CandidateEventType.PIT_STOP,
                involved_drivers=["HAM"],
                confidence=-0.1,
            )

    def test_severity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CandidateEvent(
                timestamp=NOW,
                candidate_type=CandidateEventType.RED_FLAG,
                involved_drivers=[],
                confidence=0.5,
                severity=2.0,
            )

    def test_roundtrip(self) -> None:
        ce = CandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.DRS_THREAT,
            involved_drivers=["NOR"],
            confidence=0.6,
            explanation="NOR closing fast",
            lap=15,
            session_key="2024_Monza_R",
        )
        ce2 = CandidateEvent.model_validate_json(ce.model_dump_json())
        assert ce2.candidate_type == ce.candidate_type
        assert ce2.explanation == ce.explanation


# ---- ScheduledBeat -------------------------------------------------------

class TestScheduledBeat:
    def test_create(self) -> None:
        sb = ScheduledBeat(
            timestamp=NOW,
            source_event_id="abc",
            candidate_type=CandidateEventType.LEAD_BATTLE,
            priority_score=0.9,
            involved_drivers=["HAM", "VER"],
        )
        assert sb.event_type == EventType.SCHEDULED_BEAT
        assert sb.deferred is False

    def test_roundtrip(self) -> None:
        sb = ScheduledBeat(
            timestamp=NOW,
            source_event_id="x",
            candidate_type=CandidateEventType.SAFETY_CAR,
            priority_score=1.0,
            involved_drivers=[],
            race_phase="late",
            deferred=True,
            suppress_reason="duplicate",
        )
        sb2 = ScheduledBeat.model_validate_json(sb.model_dump_json())
        assert sb2.suppress_reason == "duplicate"


# ---- RetrievalResult -----------------------------------------------------

class TestRetrievalResult:
    def test_create(self) -> None:
        rr = RetrievalResult(
            timestamp=NOW,
            query_event_id="q1",
            tier=1,
            chunks=[{"text": "some chunk", "score": 0.9}],
            latency_ms=42.0,
        )
        assert rr.event_type == EventType.RETRIEVAL_RESULT
        assert len(rr.chunks) == 1

    def test_roundtrip(self) -> None:
        rr = RetrievalResult(timestamp=NOW, query_event_id="q", tier=2)
        rr2 = RetrievalResult.model_validate_json(rr.model_dump_json())
        assert rr2.tier == 2


# ---- GeneratedCommentary -------------------------------------------------

class TestGeneratedCommentary:
    def test_create(self) -> None:
        gc = GeneratedCommentary(
            timestamp=NOW,
            source_beat_id="beat1",
            prompt_text="prompt",
            generated_text="Hamilton lunges!",
        )
        assert gc.event_type == EventType.GENERATED_COMMENTARY

    def test_roundtrip(self) -> None:
        gc = GeneratedCommentary(
            timestamp=NOW,
            source_beat_id="b",
            prompt_text="p",
            generated_text="t",
            model_variant="gpt2",
            involved_drivers=["HAM"],
        )
        gc2 = GeneratedCommentary.model_validate_json(gc.model_dump_json())
        assert gc2.model_variant == "gpt2"


# ---- GuardedCommentary ---------------------------------------------------

class TestGuardedCommentary:
    def test_create(self) -> None:
        gc = GuardedCommentary(
            timestamp=NOW,
            source_commentary_id="c1",
            original_text="original",
            adjusted_text="adjusted",
            claims=[
                ClaimSupport(claim="HAM overtook", support_type="observed", confidence=0.9)
            ],
            confidence_band="high",
        )
        assert gc.event_type == EventType.GUARDED_COMMENTARY
        assert gc.claims[0].support_type == "observed"

    def test_roundtrip(self) -> None:
        gc = GuardedCommentary(
            timestamp=NOW,
            source_commentary_id="c",
            original_text="o",
            adjusted_text="a",
        )
        gc2 = GuardedCommentary.model_validate_json(gc.model_dump_json())
        assert gc2.adjusted_text == "a"


# ---- FinalCommentary -----------------------------------------------------

class TestFinalCommentary:
    def test_create(self) -> None:
        fc = FinalCommentary(
            timestamp=NOW,
            final_text="And Hamilton wins!",
            involved_drivers=["HAM"],
        )
        assert fc.event_type == EventType.FINAL_COMMENTARY

    def test_roundtrip(self) -> None:
        fc = FinalCommentary(
            timestamp=NOW,
            final_text="text",
            storyline_id="story1",
            confidence_band="low",
            replay_lag_ms=150.0,
            race_state_summary={"lap": 57},
        )
        fc2 = FinalCommentary.model_validate_json(fc.model_dump_json())
        assert fc2.storyline_id == "story1"
        assert fc2.race_state_summary == {"lap": 57}


# ---- BaseEvent auto-fields -----------------------------------------------

class TestBaseEventDefaults:
    def test_event_id_is_uuid(self) -> None:
        tf = TelemetryFrame(timestamp=NOW, session_key="s", lap=1, drivers=[])
        import uuid as _uuid

        _uuid.UUID(tf.event_id)  # should not raise

    def test_distinct_ids(self) -> None:
        a = TelemetryFrame(timestamp=NOW, session_key="s", lap=1, drivers=[])
        b = TelemetryFrame(timestamp=NOW, session_key="s", lap=1, drivers=[])
        assert a.event_id != b.event_id
