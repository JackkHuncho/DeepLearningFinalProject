"""Comprehensive tests for Phase 11: Narrative beat manager and final output.

Coverage
========
- FinalCommentary schema validation and roundtrip
- ProgressionStage, FinalSuppressionReason, ConfidenceBand enums
- RaceStateSummary construction
- StorylineMemory: refresh blocking, semantic duplicate detection,
  progression computation, meaningful advancement
- BeatManager: process, process_batch, fallback, suppression, confidence
- TerminalAdapter: active/suppressed output, summary
- SSEAdapter: payload format, suppressed filtering, batch
- Race control handling (no drivers)
- Storyline re-narration after meaningful advancement
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

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
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary,
    GenerationConfig,
)
from src.telemetry_to_narrative.schemas.final_commentary import (
    ConfidenceBand,
    FinalCommentary,
    FinalSuppressionReason,
    ProgressionStage,
    RaceStateSummary,
)
from src.telemetry_to_narrative.beat_manager.storyline_memory import (
    DEFAULT_REFRESH_INTERVAL,
    DUPLICATE_SIMILARITY_THRESHOLD,
    StorylineMemory,
    StorylineState,
)
from src.telemetry_to_narrative.beat_manager.beat_manager import (
    BeatManager,
    build_race_state_summary,
    _compute_confidence,
)
from src.telemetry_to_narrative.beat_manager.output_adapters import (
    SSEAdapter,
    TerminalAdapter,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
SESSION_INFO = SessionInfo(year=2024, grand_prix="Monza", session_type="R")


def _make_snapshot(
    frame_id: int = 5,
    ts_offset_sec: int = 50,
    gap_sec: float = 0.8,
    intensity: float = 0.75,
    leader: str = "VER",
    safety_car: bool = False,
    track_status: str = "GREEN FLAG",
) -> RaceStateSnapshot:
    ts = BASE_TS + timedelta(seconds=ts_offset_sec)
    return RaceStateSnapshot(
        frame_id=frame_id,
        timestamp=ts,
        session_info=SESSION_INFO,
        global_state=GlobalRaceStateSummary(
            track_status=track_status,
            safety_car=safety_car,
        ),
        drivers={
            "VER": DriverRaceState(
                driver_number="1", abbreviation="VER", position=1,
                lap_number=12, gap_ahead_sec=0.0, tire_compound="MEDIUM",
                tire_age_laps=12, speed_kph=310.0,
            ),
            "HAM": DriverRaceState(
                driver_number="44", abbreviation="HAM", position=2,
                lap_number=12, gap_ahead_sec=gap_sec, tire_compound="MEDIUM",
                tire_age_laps=12, speed_kph=308.0,
            ),
        },
        active_battles=[
            BattleState(
                driver_ahead="VER", driver_behind="HAM",
                gap_sec=gap_sec, pace_delta_sec=-0.1,
                battle_intensity=intensity, battle_duration_laps=5,
            ),
        ],
        derived_summary=DerivedSummary(
            leader=leader, total_drivers=2, active_battles=1,
        ),
    )


def _make_beat(
    frame_id: int = 5,
    event_type: EventType = EventType.LEAD_BATTLE,
    drivers: list[str] | None = None,
    selected: bool = True,
    budget: int = 150,
    storyline_id: str = "battle_HAM_VER",
) -> ScheduledBeat:
    ts = BASE_TS + timedelta(seconds=frame_id * 10)
    return ScheduledBeat(
        beat_id=f"beat_{frame_id}",
        timestamp=ts,
        event_type=event_type,
        involved_drivers=drivers if drivers is not None else ["VER", "HAM"],
        storyline_id=storyline_id,
        priority_score=0.75,
        score_breakdown=ScoreBreakdown(),
        selected_for_generation=selected,
        suppression_reason=SuppressionReason.NONE if selected else SuppressionReason.LOW_PRIORITY,
        source_frame_id=frame_id,
        description="Test beat",
        output_length_budget=budget,
    )


def _make_commentary(
    beat_id: str = "beat_5",
    text: str = "Verstappen holds off Hamilton through the chicane.",
    ts_offset_sec: int = 51,
    storyline_id: str = "battle_HAM_VER",
    event_type: EventType = EventType.LEAD_BATTLE,
) -> GeneratedCommentary:
    return GeneratedCommentary(
        generation_id=f"gen_{beat_id}",
        timestamp=BASE_TS + timedelta(seconds=ts_offset_sec),
        beat_id=beat_id,
        storyline_id=storyline_id,
        event_type=event_type,
        involved_drivers=["VER", "HAM"],
        model_variant="mock",
        prompt_text="[Event: lead_battle | Drivers: VER, HAM]\n...",
        raw_output_text=text,
        commentary_text=text,
        output_length=len(text),
        latency_ms=5.0,
    )


def _make_final(
    text: str = "Verstappen holds off Hamilton.",
    suppressed: bool = False,
    confidence: ConfidenceBand = ConfidenceBand.HIGH,
    stage: ProgressionStage = ProgressionStage.INITIAL_INTEREST,
    event_type: EventType = EventType.LEAD_BATTLE,
    drivers: list[str] | None = None,
    storyline_id: str = "battle_HAM_VER",
    lap: int | None = 12,
) -> FinalCommentary:
    return FinalCommentary(
        final_id="test_final_001",
        timestamp=BASE_TS,
        beat_id="beat_5",
        storyline_id=storyline_id,
        event_type=event_type,
        involved_drivers=drivers if drivers is not None else ["VER", "HAM"],
        final_text=text,
        confidence_band=confidence,
        replay_lag_seconds=1.0,
        race_state_summary=RaceStateSummary(lap=lap, leader="VER"),
        suppression_applied=suppressed,
        suppression_reason=(
            FinalSuppressionReason.DUPLICATE_SEMANTICS if suppressed
            else FinalSuppressionReason.NONE
        ),
        progression_stage=stage,
        source_generation_id="gen_beat_5",
    )


# =====================================================================
# Schema tests
# =====================================================================


class TestFinalCommentarySchema:
    """Tests for FinalCommentary pydantic schema."""

    def test_roundtrip(self):
        final = _make_final()
        json_str = final.model_dump_json()
        restored = FinalCommentary.model_validate_json(json_str)
        assert restored.final_id == final.final_id
        assert restored.final_text == final.final_text
        assert restored.confidence_band == ConfidenceBand.HIGH
        assert restored.progression_stage == ProgressionStage.INITIAL_INTEREST

    def test_default_values(self):
        final = FinalCommentary(
            final_id="f1",
            timestamp=BASE_TS,
            beat_id="b1",
            event_type=EventType.LEAD_BATTLE,
            final_text="Test.",
        )
        assert final.suppression_applied is False
        assert final.suppression_reason == FinalSuppressionReason.NONE
        assert final.confidence_band == ConfidenceBand.MEDIUM
        assert final.replay_lag_seconds == 0.0
        assert final.race_state_summary is not None

    def test_suppressed_commentary(self):
        final = _make_final(suppressed=True, confidence=ConfidenceBand.SUPPRESSED)
        assert final.suppression_applied is True
        assert final.suppression_reason == FinalSuppressionReason.DUPLICATE_SEMANTICS
        assert final.confidence_band == ConfidenceBand.SUPPRESSED


class TestEnums:
    """Tests for Phase 11 enum types."""

    def test_progression_stages(self):
        stages = [s.value for s in ProgressionStage]
        assert "initial_interest" in stages
        assert "decisive_event" in stages
        assert "concluded" in stages

    def test_suppression_reasons(self):
        reasons = [r.value for r in FinalSuppressionReason]
        assert "none" in reasons
        assert "duplicate_semantics" in reasons
        assert "refresh_interval" in reasons
        assert "generation_failure" in reasons

    def test_confidence_bands(self):
        bands = [b.value for b in ConfidenceBand]
        assert "high" in bands
        assert "suppressed" in bands


class TestRaceStateSummary:
    """Tests for RaceStateSummary and build_race_state_summary."""

    def test_build_with_snapshot(self):
        snap = _make_snapshot()
        summary = build_race_state_summary(snap, ["VER", "HAM"])
        assert summary.lap == 12
        assert summary.leader == "VER"
        assert summary.flag_state == "GREEN FLAG"
        assert summary.safety_car is False
        assert "VER" in summary.involved_driver_summary
        assert "HAM" in summary.involved_driver_summary
        assert summary.battle_priority == 0.75

    def test_build_without_snapshot(self):
        summary = build_race_state_summary(None, ["VER", "HAM"])
        assert summary.lap is None
        assert summary.leader is None
        assert summary.involved_driver_summary == {}

    def test_build_with_unknown_driver(self):
        snap = _make_snapshot()
        summary = build_race_state_summary(snap, ["VER", "NOR"])
        assert "VER" in summary.involved_driver_summary
        assert "NOR" not in summary.involved_driver_summary

    def test_build_safety_car(self):
        snap = _make_snapshot(safety_car=True)
        summary = build_race_state_summary(snap, ["VER"])
        assert summary.safety_car is True

    def test_driver_summary_fields(self):
        snap = _make_snapshot()
        summary = build_race_state_summary(snap, ["HAM"])
        ham = summary.involved_driver_summary["HAM"]
        assert ham["position"] == 2
        assert ham["gap_ahead_sec"] == 0.8
        assert ham["tire_compound"] == "MEDIUM"
        assert ham["tire_age_laps"] == 12


# =====================================================================
# StorylineMemory tests
# =====================================================================


class TestStorylineState:
    """Tests for StorylineState initial values."""

    def test_initial_state(self):
        state = StorylineState("battle_HAM_VER")
        assert state.storyline_id == "battle_HAM_VER"
        assert state.narration_count == 0
        assert state.progression_stage == ProgressionStage.INITIAL_INTEREST
        assert state.last_frame_id == -999
        assert state.last_output_text == ""


class TestStorylineMemoryRefresh:
    """Tests for refresh interval blocking."""

    def test_first_narration_not_blocked(self):
        mem = StorylineMemory(refresh_interval=3)
        assert mem.is_refresh_blocked("story_1", 5) is False

    def test_blocked_within_interval(self):
        mem = StorylineMemory(refresh_interval=3)
        beat = _make_beat(frame_id=5)
        mem.record_narration("story_1", 5, beat, "text", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_refresh_blocked("story_1", 6) is True
        assert mem.is_refresh_blocked("story_1", 7) is True

    def test_unblocked_after_interval(self):
        mem = StorylineMemory(refresh_interval=3)
        beat = _make_beat(frame_id=5)
        mem.record_narration("story_1", 5, beat, "text", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_refresh_blocked("story_1", 8) is False

    def test_different_storyline_not_blocked(self):
        mem = StorylineMemory(refresh_interval=3)
        beat = _make_beat(frame_id=5)
        mem.record_narration("story_1", 5, beat, "text", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_refresh_blocked("story_2", 6) is False

    def test_custom_refresh_interval(self):
        mem = StorylineMemory(refresh_interval=1)
        beat = _make_beat(frame_id=5)
        mem.record_narration("story_1", 5, beat, "text", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_refresh_blocked("story_1", 6) is False


class TestStorylineMemoryDuplicates:
    """Tests for semantic duplicate detection."""

    def test_no_duplicate_first_narration(self):
        mem = StorylineMemory()
        assert mem.is_semantic_duplicate("story_1", "Some new text") is False

    def test_exact_duplicate(self):
        mem = StorylineMemory()
        beat = _make_beat()
        mem.record_narration("story_1", 5, beat, "Hamilton chases Verstappen.", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_semantic_duplicate("story_1", "Hamilton chases Verstappen.") is True

    def test_near_duplicate(self):
        mem = StorylineMemory()
        beat = _make_beat()
        mem.record_narration("story_1", 5, beat, "Hamilton chases Verstappen closely.", ProgressionStage.INITIAL_INTEREST)
        # Very similar text should be flagged.
        assert mem.is_semantic_duplicate("story_1", "Hamilton chases Verstappen closely!") is True

    def test_different_text_not_duplicate(self):
        mem = StorylineMemory()
        beat = _make_beat()
        mem.record_narration("story_1", 5, beat, "Hamilton chases Verstappen.", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_semantic_duplicate("story_1", "Pit stop for Leclerc, new hards fitted.") is False

    def test_case_insensitive(self):
        mem = StorylineMemory()
        beat = _make_beat()
        mem.record_narration("story_1", 5, beat, "HAMILTON CHASES VERSTAPPEN.", ProgressionStage.INITIAL_INTEREST)
        assert mem.is_semantic_duplicate("story_1", "hamilton chases verstappen.") is True


class TestStorylineMemoryProgression:
    """Tests for progression stage computation."""

    def test_first_narration_is_initial_interest(self):
        mem = StorylineMemory()
        beat = _make_beat()
        snap = _make_snapshot()
        stage = mem.compute_progression("story_1", beat, snap)
        assert stage == ProgressionStage.INITIAL_INTEREST

    def test_decisive_event_for_overtake(self):
        mem = StorylineMemory()
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST)

        beat_2 = _make_beat(frame_id=5, event_type=EventType.OVERTAKE)
        stage = mem.compute_progression("story_1", beat_2, None)
        assert stage == ProgressionStage.DECISIVE_EVENT

    def test_decisive_event_for_pit_strategy(self):
        mem = StorylineMemory()
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST)

        beat_2 = _make_beat(frame_id=5, event_type=EventType.PIT_STRATEGY)
        stage = mem.compute_progression("story_1", beat_2, None)
        assert stage == ProgressionStage.DECISIVE_EVENT

    def test_decisive_event_for_race_control(self):
        mem = StorylineMemory()
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST)

        beat_2 = _make_beat(frame_id=5, event_type=EventType.RACE_CONTROL, drivers=[])
        stage = mem.compute_progression("story_1", beat_2, None)
        assert stage == ProgressionStage.DECISIVE_EVENT

    def test_gap_closing_advances_stage(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=1.0, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.5, intensity=0.5)
        beat_2 = _make_beat(frame_id=5)
        stage = mem.compute_progression("story_1", beat_2, snap_2)
        assert stage == ProgressionStage.ACTIVE_PRESSURE

    def test_intensity_rising_advances_stage(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.7)
        beat_2 = _make_beat(frame_id=5)
        stage = mem.compute_progression("story_1", beat_2, snap_2)
        assert stage == ProgressionStage.ACTIVE_PRESSURE

    def test_no_change_holds_stage(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.ACTIVE_PRESSURE, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.5)
        beat_2 = _make_beat(frame_id=5)
        stage = mem.compute_progression("story_1", beat_2, snap_2)
        assert stage == ProgressionStage.ACTIVE_PRESSURE

    def test_advance_caps_at_decisive(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=1.0, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.DECISIVE_EVENT, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.3, intensity=0.9)
        beat_2 = _make_beat(frame_id=5)
        stage = mem.compute_progression("story_1", beat_2, snap_2)
        assert stage == ProgressionStage.DECISIVE_EVENT


class TestStorylineMemoryAdvancement:
    """Tests for has_meaningfully_advanced."""

    def test_first_narration_always_advances(self):
        mem = StorylineMemory()
        beat = _make_beat()
        assert mem.has_meaningfully_advanced("story_1", beat, None) is True

    def test_event_type_change_advances(self):
        mem = StorylineMemory()
        beat_1 = _make_beat(frame_id=1, event_type=EventType.LEAD_BATTLE)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST)

        beat_2 = _make_beat(frame_id=5, event_type=EventType.OVERTAKE)
        assert mem.has_meaningfully_advanced("story_1", beat_2, None) is True

    def test_gap_change_advances(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=1.0)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.5)
        beat_2 = _make_beat(frame_id=5)
        assert mem.has_meaningfully_advanced("story_1", beat_2, snap_2) is True

    def test_no_change_does_not_advance(self):
        mem = StorylineMemory()
        snap = _make_snapshot(frame_id=1, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST, snap)

        # Same gap, same intensity, same event type.
        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.5)
        beat_2 = _make_beat(frame_id=5)
        assert mem.has_meaningfully_advanced("story_1", beat_2, snap_2) is False

    def test_intensity_change_advances(self):
        mem = StorylineMemory()
        snap_1 = _make_snapshot(frame_id=1, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        mem.record_narration("story_1", 1, beat_1, "text", ProgressionStage.INITIAL_INTEREST, snap_1)

        snap_2 = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.7)
        beat_2 = _make_beat(frame_id=5)
        assert mem.has_meaningfully_advanced("story_1", beat_2, snap_2) is True


# =====================================================================
# BeatManager tests
# =====================================================================


class TestBeatManagerProcess:
    """Tests for BeatManager.process."""

    def test_first_beat_emitted(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=5)
        commentary = _make_commentary(beat_id="beat_5")
        snap = _make_snapshot(frame_id=5)

        result = manager.process(beat, commentary, snap)
        assert result is not None
        assert result.final_text == commentary.commentary_text
        assert result.suppression_applied is False
        assert result.storyline_id == "battle_HAM_VER"
        assert result.confidence_band in (ConfidenceBand.HIGH, ConfidenceBand.MEDIUM)

    def test_generation_failure_returns_fallback(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=5)

        result = manager.process(beat, None, None)
        assert result is not None
        assert result.suppression_applied is True
        assert result.suppression_reason == FinalSuppressionReason.GENERATION_FAILURE
        assert result.confidence_band == ConfidenceBand.LOW
        assert len(result.final_text) > 0

    def test_empty_commentary_returns_fallback(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=5)
        commentary = _make_commentary(beat_id="beat_5", text="   ")

        result = manager.process(beat, commentary, None)
        assert result is not None
        assert result.suppression_reason == FinalSuppressionReason.GENERATION_FAILURE

    def test_refresh_interval_suppresses(self):
        manager = BeatManager(refresh_interval=3)
        snap = _make_snapshot(frame_id=5)
        beat_1 = _make_beat(frame_id=5)
        commentary_1 = _make_commentary(beat_id="beat_5")
        manager.process(beat_1, commentary_1, snap)

        # Beat at frame 6 — within refresh interval.
        beat_2 = _make_beat(frame_id=6)
        commentary_2 = _make_commentary(
            beat_id="beat_6",
            text="Hamilton closing in on Verstappen.",
            ts_offset_sec=61,
        )
        result = manager.process(beat_2, commentary_2, None)
        assert result is None  # suppressed, include_suppressed=False

    def test_refresh_interval_suppresses_with_include(self):
        manager = BeatManager(refresh_interval=3, include_suppressed=True)
        snap = _make_snapshot(frame_id=5)
        beat_1 = _make_beat(frame_id=5)
        commentary_1 = _make_commentary(beat_id="beat_5")
        manager.process(beat_1, commentary_1, snap)

        beat_2 = _make_beat(frame_id=6)
        commentary_2 = _make_commentary(
            beat_id="beat_6",
            text="Hamilton closing in on Verstappen.",
            ts_offset_sec=61,
        )
        result = manager.process(beat_2, commentary_2, None)
        assert result is not None
        assert result.suppression_applied is True
        assert result.suppression_reason == FinalSuppressionReason.REFRESH_INTERVAL
        assert result.confidence_band == ConfidenceBand.SUPPRESSED

    def test_semantic_duplicate_suppresses(self):
        manager = BeatManager(refresh_interval=1)
        snap_1 = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=5)
        text = "Verstappen holds off Hamilton through the chicane."
        commentary_1 = _make_commentary(beat_id="beat_5", text=text)
        manager.process(beat_1, commentary_1, snap_1)

        # Same storyline, almost identical text, gap changed enough to pass advancement.
        snap_2 = _make_snapshot(frame_id=10, gap_sec=0.5, intensity=0.5)
        beat_2 = _make_beat(frame_id=10)
        commentary_2 = _make_commentary(
            beat_id="beat_10",
            text="Verstappen holds off Hamilton through the chicane!",
            ts_offset_sec=100,
        )
        result = manager.process(beat_2, commentary_2, snap_2)
        assert result is None

    def test_no_progression_suppresses(self):
        manager = BeatManager(refresh_interval=1)
        snap = _make_snapshot(frame_id=5, gap_sec=0.8, intensity=0.5)
        beat_1 = _make_beat(frame_id=5)
        commentary_1 = _make_commentary(beat_id="beat_5")
        manager.process(beat_1, commentary_1, snap)

        # Same gap, same intensity, same event type — no meaningful advancement.
        snap_2 = _make_snapshot(frame_id=10, gap_sec=0.8, intensity=0.5)
        beat_2 = _make_beat(frame_id=10)
        commentary_2 = _make_commentary(
            beat_id="beat_10",
            text="A completely different commentary text for variety.",
            ts_offset_sec=100,
        )
        result = manager.process(beat_2, commentary_2, snap_2)
        assert result is None

    def test_advancement_allows_renarration(self):
        """After gap closes significantly, same storyline can be narrated again."""
        manager = BeatManager(refresh_interval=1)
        snap_1 = _make_snapshot(frame_id=5, gap_sec=1.0, intensity=0.5)
        beat_1 = _make_beat(frame_id=5)
        commentary_1 = _make_commentary(beat_id="beat_5", text="The gap is 1.0 seconds.")
        manager.process(beat_1, commentary_1, snap_1)

        # Gap has closed — advancement detected.
        snap_2 = _make_snapshot(frame_id=10, gap_sec=0.4, intensity=0.7)
        beat_2 = _make_beat(frame_id=10)
        commentary_2 = _make_commentary(
            beat_id="beat_10",
            text="Hamilton is right behind Verstappen now!",
            ts_offset_sec=100,
        )
        result = manager.process(beat_2, commentary_2, snap_2)
        assert result is not None
        assert result.suppression_applied is False

    def test_replay_lag_computed(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=5)  # ts = BASE_TS + 50s
        commentary = _make_commentary(beat_id="beat_5", ts_offset_sec=52)
        result = manager.process(beat, commentary)
        assert result is not None
        assert result.replay_lag_seconds >= 0.0

    def test_fallback_text_per_event_type(self):
        manager = BeatManager()
        for event_type in [EventType.LEAD_BATTLE, EventType.OVERTAKE, EventType.PIT_STRATEGY]:
            beat = _make_beat(frame_id=5, event_type=event_type)
            result = manager.process(beat, None, None)
            assert result is not None
            assert len(result.final_text) > 5

    def test_race_control_no_drivers(self):
        manager = BeatManager()
        beat = _make_beat(
            frame_id=5,
            event_type=EventType.RACE_CONTROL,
            drivers=[],
            storyline_id="race_control",
        )
        commentary = _make_commentary(
            beat_id="beat_5",
            text="Safety car deployed on lap 12.",
            storyline_id="race_control",
            event_type=EventType.RACE_CONTROL,
        )
        result = manager.process(beat, commentary)
        assert result is not None
        assert result.involved_drivers == []
        assert result.storyline_id == "race_control"


class TestBeatManagerProcessBatch:
    """Tests for BeatManager.process_batch."""

    def test_batch_filters_unselected(self):
        manager = BeatManager()
        beats = [
            _make_beat(frame_id=1, selected=True),
            _make_beat(frame_id=2, selected=False),
            _make_beat(frame_id=3, selected=True),
        ]
        commentaries = [
            _make_commentary(beat_id="beat_1", text="Commentary 1.", ts_offset_sec=11),
            _make_commentary(beat_id="beat_3", text="Commentary 3.", ts_offset_sec=31),
        ]
        # Different storylines so no suppression.
        beats[0].storyline_id = "story_a"
        beats[2].storyline_id = "story_b"
        commentaries[0].storyline_id = "story_a"
        commentaries[1].storyline_id = "story_b"

        results = manager.process_batch(beats, commentaries)
        # Only selected beats processed; both should emit since different storylines.
        assert len(results) == 2

    def test_batch_matches_by_beat_id(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=7)
        commentary = _make_commentary(beat_id="beat_7", ts_offset_sec=71)

        results = manager.process_batch([beat], [commentary])
        assert len(results) == 1
        assert results[0].beat_id == "beat_7"

    def test_batch_missing_commentary_falls_back(self):
        manager = BeatManager()
        beat = _make_beat(frame_id=7)
        # No commentary for beat_7.
        results = manager.process_batch([beat], [])
        assert len(results) == 1
        assert results[0].suppression_reason == FinalSuppressionReason.GENERATION_FAILURE

    def test_batch_with_snapshots(self):
        manager = BeatManager()
        snap = _make_snapshot(frame_id=5)
        beat = _make_beat(frame_id=5)
        commentary = _make_commentary(beat_id="beat_5")

        results = manager.process_batch([beat], [commentary], [snap])
        assert len(results) == 1
        assert results[0].race_state_summary.lap == 12


class TestConfidenceComputation:
    """Tests for _compute_confidence."""

    def test_fallback_is_low(self):
        assert _compute_confidence(ProgressionStage.INITIAL_INTEREST, "text", True) == ConfidenceBand.LOW

    def test_decisive_event_is_high(self):
        assert _compute_confidence(ProgressionStage.DECISIVE_EVENT, "A long enough text.", False) == ConfidenceBand.HIGH

    def test_active_pressure_long_text_is_high(self):
        text = "Hamilton is closing rapidly on Verstappen through sector three."
        assert _compute_confidence(ProgressionStage.ACTIVE_PRESSURE, text, False) == ConfidenceBand.HIGH

    def test_active_pressure_short_text_is_medium(self):
        assert _compute_confidence(ProgressionStage.ACTIVE_PRESSURE, "Short.", False) == ConfidenceBand.MEDIUM

    def test_initial_interest_is_medium(self):
        assert _compute_confidence(ProgressionStage.INITIAL_INTEREST, "A longer text.", False) == ConfidenceBand.MEDIUM


# =====================================================================
# Output adapter tests
# =====================================================================


class TestTerminalAdapter:
    """Tests for TerminalAdapter."""

    def test_emit_active_commentary(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        final = _make_final()

        adapter.emit(final)
        output = buf.getvalue()
        assert "lead_battle" in output
        assert "Verstappen holds off Hamilton." in output
        assert "battle_HAM_VER" in output

    def test_emit_suppressed_hidden_by_default(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False, show_suppressed=False)
        final = _make_final(suppressed=True)

        adapter.emit(final)
        assert buf.getvalue() == ""

    def test_emit_suppressed_shown_when_enabled(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False, show_suppressed=True)
        final = _make_final(suppressed=True)

        adapter.emit(final)
        output = buf.getvalue()
        assert "SUPPRESSED" in output

    def test_emit_batch(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        finals = [
            _make_final(text="Commentary one."),
            _make_final(text="Commentary two.", suppressed=True),
            _make_final(text="Commentary three."),
        ]
        # Different final_ids to avoid confusion.
        finals[2].storyline_id = "story_b"

        adapter.emit_batch(finals)
        output = buf.getvalue()
        assert "Commentary one." in output
        assert "Commentary two." not in output  # suppressed
        assert "Commentary three." in output

    def test_summary(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        adapter.emit(_make_final())
        adapter.emit(_make_final(suppressed=True))
        adapter.print_summary()
        output = buf.getvalue()
        assert "1 emitted" in output
        assert "1 suppressed" in output

    def test_colour_mode(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=True)
        adapter.emit(_make_final())
        output = buf.getvalue()
        # ANSI escape codes should be present.
        assert "\033[" in output

    def test_no_colour_mode(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        adapter.emit(_make_final())
        output = buf.getvalue()
        assert "\033[" not in output

    def test_lap_displayed(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        adapter.emit(_make_final(lap=42))
        output = buf.getvalue()
        assert "L42" in output

    def test_no_drivers_shows_field(self):
        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        adapter.emit(_make_final(drivers=[]))
        output = buf.getvalue()
        assert "field" in output


class TestSSEAdapter:
    """Tests for SSEAdapter."""

    def test_format_active(self):
        adapter = SSEAdapter()
        final = _make_final()
        payload = adapter.format(final)
        assert payload is not None
        assert payload.startswith("event: commentary\n")
        assert "data: " in payload
        assert payload.endswith("\n\n")

        # Parse data line.
        data_line = payload.split("data: ", 1)[1].strip()
        data = json.loads(data_line)
        assert data["final_id"] == "test_final_001"
        assert data["final_text"] == "Verstappen holds off Hamilton."
        assert data["event_type"] == "lead_battle"
        assert data["confidence_band"] == "high"
        assert data["suppression_applied"] is False

    def test_format_suppressed_hidden_by_default(self):
        adapter = SSEAdapter(include_suppressed=False)
        final = _make_final(suppressed=True)
        assert adapter.format(final) is None

    def test_format_suppressed_when_included(self):
        adapter = SSEAdapter(include_suppressed=True)
        final = _make_final(suppressed=True)
        payload = adapter.format(final)
        assert payload is not None
        assert "event: suppressed" in payload

    def test_format_batch(self):
        adapter = SSEAdapter()
        finals = [
            _make_final(text="One"),
            _make_final(text="Two", suppressed=True),
            _make_final(text="Three"),
        ]
        payloads = adapter.format_batch(finals)
        assert len(payloads) == 2  # suppressed one excluded

    def test_emit_to_stream(self):
        buf = io.StringIO()
        adapter = SSEAdapter()
        final = _make_final()
        emitted = adapter.emit(final, stream=buf)
        assert emitted is True
        assert "event: commentary" in buf.getvalue()

    def test_emit_suppressed_returns_false(self):
        buf = io.StringIO()
        adapter = SSEAdapter(include_suppressed=False)
        final = _make_final(suppressed=True)
        emitted = adapter.emit(final, stream=buf)
        assert emitted is False
        assert buf.getvalue() == ""

    def test_payload_has_race_state(self):
        adapter = SSEAdapter()
        final = _make_final(lap=15)
        payload = adapter.format(final)
        data = json.loads(payload.split("data: ", 1)[1].strip())
        assert data["race_state"]["lap"] == 15
        assert data["race_state"]["leader"] == "VER"

    def test_payload_has_progression(self):
        adapter = SSEAdapter()
        final = _make_final(stage=ProgressionStage.DECISIVE_EVENT)
        payload = adapter.format(final)
        data = json.loads(payload.split("data: ", 1)[1].strip())
        assert data["progression_stage"] == "decisive_event"

    def test_payload_has_storyline(self):
        adapter = SSEAdapter()
        final = _make_final(storyline_id="battle_LEC_NOR")
        payload = adapter.format(final)
        data = json.loads(payload.split("data: ", 1)[1].strip())
        assert data["storyline_id"] == "battle_LEC_NOR"


# =====================================================================
# Integration-style tests
# =====================================================================


class TestEndToEnd:
    """Integration-style tests combining multiple components."""

    def test_full_pipeline_three_beats(self):
        """Process 3 beats for the same storyline across frames."""
        manager = BeatManager(refresh_interval=3)

        # Beat 1 — initial.
        snap_1 = _make_snapshot(frame_id=1, gap_sec=1.0, intensity=0.5)
        beat_1 = _make_beat(frame_id=1)
        com_1 = _make_commentary(beat_id="beat_1", text="VER leads HAM by one second.", ts_offset_sec=11)
        r1 = manager.process(beat_1, com_1, snap_1)
        assert r1 is not None
        assert r1.progression_stage == ProgressionStage.INITIAL_INTEREST

        # Beat 2 — too soon (frame 3, within refresh interval of 3).
        beat_2 = _make_beat(frame_id=3)
        com_2 = _make_commentary(beat_id="beat_3", text="HAM is closing the gap.", ts_offset_sec=31)
        r2 = manager.process(beat_2, com_2, None)
        assert r2 is None  # suppressed

        # Beat 3 — enough frames later, gap has closed.
        snap_3 = _make_snapshot(frame_id=10, gap_sec=0.4, intensity=0.8)
        beat_3 = _make_beat(frame_id=10)
        com_3 = _make_commentary(
            beat_id="beat_10",
            text="Hamilton right on the tail of Verstappen!",
            ts_offset_sec=101,
        )
        r3 = manager.process(beat_3, com_3, snap_3)
        assert r3 is not None
        assert r3.suppression_applied is False
        assert r3.progression_stage in (
            ProgressionStage.ACTIVE_PRESSURE,
            ProgressionStage.IMMINENT_ACTION,
        )

    def test_multiple_storylines_independent(self):
        """Different storylines don't interfere with each other."""
        manager = BeatManager(refresh_interval=3)

        beat_a = _make_beat(frame_id=5, storyline_id="story_a")
        com_a = _make_commentary(beat_id="beat_5", text="Story A commentary.", storyline_id="story_a")

        beat_b = _make_beat(frame_id=5, storyline_id="story_b")
        beat_b.beat_id = "beat_5b"
        com_b = _make_commentary(beat_id="beat_5b", text="Story B commentary.", storyline_id="story_b")

        r_a = manager.process(beat_a, com_a)
        r_b = manager.process(beat_b, com_b)

        assert r_a is not None
        assert r_b is not None
        assert r_a.storyline_id == "story_a"
        assert r_b.storyline_id == "story_b"

    def test_terminal_and_sse_produce_output(self):
        """Both adapters produce output for the same FinalCommentary."""
        manager = BeatManager()
        beat = _make_beat(frame_id=5)
        com = _make_commentary(beat_id="beat_5")
        final = manager.process(beat, com, _make_snapshot(frame_id=5))
        assert final is not None

        # Terminal.
        term_buf = io.StringIO()
        TerminalAdapter(stream=term_buf, colour=False).emit(final)
        assert len(term_buf.getvalue()) > 0

        # SSE.
        sse_payload = SSEAdapter().format(final)
        assert sse_payload is not None
        data = json.loads(sse_payload.split("data: ", 1)[1].strip())
        assert data["final_text"] == final.final_text

    def test_batch_then_terminal_output(self):
        manager = BeatManager(refresh_interval=1)
        beats = [
            _make_beat(frame_id=1, storyline_id="s1"),
            _make_beat(frame_id=5, storyline_id="s2"),
        ]
        beats[1].beat_id = "beat_5"
        coms = [
            _make_commentary(beat_id="beat_1", text="First storyline.", ts_offset_sec=11, storyline_id="s1"),
            _make_commentary(beat_id="beat_5", text="Second storyline.", ts_offset_sec=51, storyline_id="s2"),
        ]
        snaps = [_make_snapshot(frame_id=1), _make_snapshot(frame_id=5)]

        finals = manager.process_batch(beats, coms, snaps)
        assert len(finals) == 2

        buf = io.StringIO()
        adapter = TerminalAdapter(stream=buf, colour=False)
        adapter.emit_batch(finals)
        adapter.print_summary()
        output = buf.getvalue()
        assert "2 emitted" in output
