"""Tests for Phase 5: EditorialScheduler, scoring, storylines, backpressure.

Covers:
1. Precedence ordering
2. Suppression logic (threshold, cooldown, storyline-not-advanced, duplicate)
3. Storyline continuation
4. Threshold behaviour
5. Backpressure / load-shedding behaviour
6. Deterministic scheduling decisions on synthetic scenarios
7. Schema validity
8. Snapshot-to-candidate adapter
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telemetry_to_narrative.schemas.candidate_event import (
    CandidateEvent,
    EventType,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScoreBreakdown,
    ScheduledBeat,
    SuppressionReason,
)
from src.telemetry_to_narrative.schemas.race_state import (
    BattleState,
    DerivedSummary,
    DriverRaceState,
    GlobalRaceStateSummary,
    RaceStateSnapshot,
)
from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo
from src.telemetry_to_narrative.config.scheduler_config import SchedulerConfig
from src.telemetry_to_narrative.scheduler.editorial_scheduler import (
    EditorialScheduler,
    SchedulerCounters,
    StorylineTracker,
    _derive_storyline_id,
    apply_backpressure_adjustments,
    score_contextual_importance,
    score_race_phase,
    score_storyline_relevance,
    score_urgency,
)
from src.telemetry_to_narrative.scheduler.snapshot_to_candidates import (
    extract_candidates,
)


# ── Helpers ───────────────────────────────────────────────────────────────

TS = datetime(2024, 9, 1, 14, 30, 0, tzinfo=timezone.utc)
SESSION = SessionInfo(year=2024, grand_prix="Test GP", session_type="Race")


def _evt(
    etype: EventType = EventType.MIDFIELD_BATTLE,
    drivers: list[str] | None = None,
    confidence: float = 1.0,
    frame: int = 0,
    gap: float | None = None,
    intensity: float | None = None,
    safety: bool = False,
    position: int | None = None,
    desc: str | None = None,
) -> CandidateEvent:
    return CandidateEvent(
        event_id=f"evt_{etype.value}_{frame}",
        timestamp=TS + timedelta(seconds=frame * 85),
        event_type=etype,
        involved_drivers=drivers or [],
        confidence=confidence,
        source_frame_id=frame,
        description=desc or f"{etype.value} at frame {frame}",
        position_of_interest=position,
        gap_sec=gap,
        battle_intensity=intensity,
        safety_critical=safety,
    )


def _snap(
    frame_id: int = 0,
    track_status: str = "GREEN FLAG",
    safety_car: bool = False,
    battles: list[BattleState] | None = None,
    drivers: dict[str, DriverRaceState] | None = None,
) -> RaceStateSnapshot:
    if drivers is None:
        drivers = {
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1),
            "HAM": DriverRaceState(driver_number="44", abbreviation="HAM", position=2, gap_ahead_sec=0.8),
        }
    return RaceStateSnapshot(
        frame_id=frame_id,
        timestamp=TS + timedelta(seconds=frame_id * 85),
        session_info=SESSION,
        global_state=GlobalRaceStateSummary(
            track_status=track_status,
            safety_car=safety_car,
        ),
        drivers=drivers,
        active_battles=battles or [],
        derived_summary=DerivedSummary(leader="VER", total_drivers=len(drivers)),
    )


# ── 1. Precedence ordering ───────────────────────────────────────────────

class TestPrecedenceOrdering:
    def test_safety_beats_lead_battle(self):
        """Race control should always rank above a lead battle."""
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5, intensity=0.9, position=1),
            _evt(EventType.RACE_CONTROL, safety=True, desc="SAFETY CAR"),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.RACE_CONTROL

    def test_lead_battle_beats_midfield(self):
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.MIDFIELD_BATTLE, ["SAI", "NOR"], gap=0.7, intensity=0.6, position=6),
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5, intensity=0.8, position=1),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.LEAD_BATTLE

    def test_podium_battle_beats_telemetry(self):
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.TELEMETRY_CHANGE, ["LEC"], position=5),
            _evt(EventType.PODIUM_BATTLE, ["HAM", "LEC"], gap=0.6, intensity=0.7, position=2),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.PODIUM_BATTLE

    def test_pit_strategy_beats_telemetry(self):
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.TELEMETRY_CHANGE, position=10),
            _evt(EventType.PIT_STRATEGY, ["VER"], position=1, desc="VER pits SOFT→HARD"),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.PIT_STRATEGY

    def test_only_one_beat_selected_per_frame(self):
        """Even with many high-scoring events, only one is selected."""
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.RACE_CONTROL, safety=True),
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.3, intensity=0.95, position=1),
            _evt(EventType.PODIUM_BATTLE, ["HAM", "LEC"], gap=0.5, intensity=0.8, position=2),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1


# ── 2. Suppression logic ─────────────────────────────────────────────────

class TestSuppressionLogic:
    def test_low_priority_suppressed(self):
        """Telemetry changes should score below the narration threshold."""
        config = SchedulerConfig(narration_threshold=0.60)
        scheduler = EditorialScheduler(config)
        candidates = [_evt(EventType.TELEMETRY_CHANGE, confidence=0.5, position=15)]
        beats = scheduler.schedule(candidates, frame_id=0)
        assert not beats[0].selected_for_generation
        assert beats[0].suppression_reason == SuppressionReason.LOW_PRIORITY

    def test_cooldown_suppression(self):
        """Same storyline narrated within cooldown window is suppressed."""
        config = SchedulerConfig()
        config.storyline.cooldown_frames = 3
        scheduler = EditorialScheduler(config)

        lead = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5, intensity=0.9, position=1)

        # Frame 0: should be selected.
        beats0 = scheduler.schedule([lead], frame_id=0)
        assert beats0[0].selected_for_generation

        # Frame 1: same storyline, within cooldown.
        lead1 = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.4, intensity=0.9, position=1, frame=1)
        beats1 = scheduler.schedule([lead1], frame_id=1)
        assert not beats1[0].selected_for_generation
        assert beats1[0].suppression_reason == SuppressionReason.COOLDOWN

    def test_storyline_not_advanced_suppression(self):
        """Battle storyline that hasn't meaningfully changed is suppressed."""
        config = SchedulerConfig()
        config.storyline.cooldown_frames = 1  # short cooldown so cooldown isn't the reason
        config.storyline.battle_advance_threshold_sec = 0.5
        scheduler = EditorialScheduler(config)

        # Frame 0: narrate the battle.
        lead0 = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=1.0, intensity=0.8, position=1, frame=0)
        beats0 = scheduler.schedule([lead0], frame_id=0)
        assert beats0[0].selected_for_generation

        # Frame 2: same gap (change < 0.5s threshold).
        lead2 = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=1.1, intensity=0.8, position=1, frame=2)
        beats2 = scheduler.schedule([lead2], frame_id=2)
        assert not beats2[0].selected_for_generation
        assert beats2[0].suppression_reason == SuppressionReason.STORYLINE_NOT_ADVANCED

    def test_backpressure_drop_suppression(self):
        """Non-critical events dropped under extreme backpressure."""
        config = SchedulerConfig()
        scheduler = EditorialScheduler(config)
        scheduler.queue_depth = config.backpressure.queue_hard_limit + 2

        candidates = [_evt(EventType.MIDFIELD_BATTLE, ["SAI", "NOR"], gap=0.8, intensity=0.6, position=6)]
        beats = scheduler.schedule(candidates, frame_id=0)
        assert not beats[0].selected_for_generation
        assert beats[0].suppression_reason == SuppressionReason.BACKPRESSURE_DROP

    def test_safety_event_survives_backpressure(self):
        """Safety-critical events are NOT dropped even under hard backpressure."""
        config = SchedulerConfig()
        scheduler = EditorialScheduler(config)
        scheduler.queue_depth = config.backpressure.queue_hard_limit + 5

        candidates = [_evt(EventType.RACE_CONTROL, safety=True)]
        beats = scheduler.schedule(candidates, frame_id=0)
        assert beats[0].selected_for_generation


# ── 3. Storyline continuation ────────────────────────────────────────────

class TestStorylineContinuation:
    def test_storyline_id_consistent_for_same_pair(self):
        e1 = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"])
        e2 = _evt(EventType.LEAD_BATTLE, ["HAM", "VER"])  # order reversed
        assert _derive_storyline_id(e1) == _derive_storyline_id(e2)

    def test_storyline_id_differs_for_different_types(self):
        e1 = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"])
        e2 = _evt(EventType.PIT_STRATEGY, ["VER"])
        assert _derive_storyline_id(e1) != _derive_storyline_id(e2)

    def test_race_control_shares_single_storyline(self):
        e1 = _evt(EventType.RACE_CONTROL, safety=True, desc="SC deployed")
        e2 = _evt(EventType.RACE_CONTROL, safety=True, desc="SC ending")
        assert _derive_storyline_id(e1) == _derive_storyline_id(e2) == "race_control"

    def test_advanced_battle_can_be_re_narrated(self):
        """If gap changes enough, the same battle should pass the advance check."""
        config = SchedulerConfig()
        config.storyline.cooldown_frames = 1
        config.storyline.battle_advance_threshold_sec = 0.3
        scheduler = EditorialScheduler(config)

        # Frame 0: gap 1.0
        beats0 = scheduler.schedule(
            [_evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=1.0, intensity=0.8, position=1, frame=0)],
            frame_id=0,
        )
        assert beats0[0].selected_for_generation

        # Frame 2: gap narrowed significantly (0.4 change > 0.3 threshold)
        beats2 = scheduler.schedule(
            [_evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.6, intensity=0.9, position=1, frame=2)],
            frame_id=2,
        )
        assert beats2[0].selected_for_generation

    def test_tracker_records_narrations(self):
        config = SchedulerConfig()
        tracker = StorylineTracker(config)
        tracker.record_narration("battle_HAM_VER", frame_id=5, gap_sec=0.8)
        state = tracker._storylines["battle_HAM_VER"]
        assert state.last_narrated_frame == 5
        assert state.last_gap_sec == 0.8
        assert state.narration_count == 1


# ── 4. Threshold behaviour ───────────────────────────────────────────────

class TestThresholdBehaviour:
    def test_raising_threshold_suppresses_more(self):
        """Higher threshold → fewer beats selected."""
        low = SchedulerConfig(narration_threshold=0.20)
        high = SchedulerConfig(narration_threshold=0.80)

        candidates = [
            _evt(EventType.MIDFIELD_BATTLE, ["SAI", "NOR"], gap=0.8, intensity=0.5, position=6),
        ]

        beats_low = EditorialScheduler(low).schedule(candidates, frame_id=0)
        beats_high = EditorialScheduler(high).schedule(candidates, frame_id=0)

        selected_low = sum(1 for b in beats_low if b.selected_for_generation)
        selected_high = sum(1 for b in beats_high if b.selected_for_generation)
        assert selected_low >= selected_high

    def test_zero_threshold_selects_everything(self):
        config = SchedulerConfig(narration_threshold=0.0)
        scheduler = EditorialScheduler(config)
        candidates = [_evt(EventType.TELEMETRY_CHANGE, confidence=0.3)]
        beats = scheduler.schedule(candidates, frame_id=0)
        assert beats[0].selected_for_generation

    def test_score_above_threshold_selected(self):
        config = SchedulerConfig(narration_threshold=0.35)
        scheduler = EditorialScheduler(config)
        candidates = [_evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.3, intensity=0.9, position=1)]
        beats = scheduler.schedule(candidates, frame_id=0)
        assert beats[0].selected_for_generation
        assert beats[0].priority_score >= 0.35


# ── 5. Backpressure / load-shedding ──────────────────────────────────────

class TestBackpressure:
    def test_penalty_increases_with_queue_depth(self):
        config = SchedulerConfig()
        _, _, _ = apply_backpressure_adjustments(config, queue_depth=0, latency_estimate_ms=0)
        p1, _, _ = apply_backpressure_adjustments(config, queue_depth=config.backpressure.queue_soft_limit + 2, latency_estimate_ms=0)
        p2, _, _ = apply_backpressure_adjustments(config, queue_depth=config.backpressure.queue_soft_limit + 5, latency_estimate_ms=0)
        assert p2 > p1 > 0

    def test_retrieval_disabled_at_cutoff(self):
        config = SchedulerConfig()
        _, ret_ok, _ = apply_backpressure_adjustments(config, queue_depth=0, latency_estimate_ms=0)
        assert ret_ok is True

        _, ret_no, _ = apply_backpressure_adjustments(config, queue_depth=config.backpressure.queue_retrieval_cutoff, latency_estimate_ms=0)
        assert ret_no is False

    def test_retrieval_disabled_by_latency(self):
        config = SchedulerConfig()
        _, ret_no, _ = apply_backpressure_adjustments(
            config, queue_depth=0,
            latency_estimate_ms=config.backpressure.latency_retrieval_cutoff_ms + 100,
        )
        assert ret_no is False

    def test_length_budget_reduces_under_load(self):
        config = SchedulerConfig()
        _, _, budget_normal = apply_backpressure_adjustments(config, queue_depth=0, latency_estimate_ms=0)
        _, _, budget_moderate = apply_backpressure_adjustments(config, queue_depth=config.backpressure.queue_soft_limit + 1, latency_estimate_ms=0)
        _, _, budget_heavy = apply_backpressure_adjustments(config, queue_depth=config.backpressure.queue_hard_limit, latency_estimate_ms=0)

        assert budget_normal > budget_moderate > budget_heavy

    def test_beat_reflects_backpressure_state(self):
        config = SchedulerConfig()
        scheduler = EditorialScheduler(config)
        scheduler.queue_depth = config.backpressure.queue_retrieval_cutoff + 1

        candidates = [_evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.3, intensity=0.9, position=1)]
        beats = scheduler.schedule(candidates, frame_id=0)

        assert not beats[0].retrieval_allowed
        assert beats[0].output_length_budget < config.default_length_budget

    def test_counters_track_retrieval_disabled(self):
        config = SchedulerConfig()
        scheduler = EditorialScheduler(config)
        scheduler.queue_depth = config.backpressure.queue_retrieval_cutoff + 1

        scheduler.schedule(
            [_evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.3, intensity=0.9, position=1)],
            frame_id=0,
        )
        assert scheduler.counters.retrieval_disabled_count >= 1


# ── 6. Deterministic synthetic scenarios ──────────────────────────────────

class TestSyntheticScenarios:
    def test_simultaneous_lead_battle_and_yellow_flag(self):
        """Yellow flag (race control) must take priority over lead battle."""
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5, intensity=0.9, position=1),
            _evt(EventType.RACE_CONTROL, safety=True, desc="YELLOW FLAG sector 2"),
        ]
        beats = scheduler.schedule(candidates, frame_id=0, safety_car_active=True)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.RACE_CONTROL

    def test_lead_battle_plus_midfield_overtake(self):
        """Lead battle should be selected over midfield overtake."""
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.OVERTAKE, ["SAI"], position=8, desc="SAI gains P8"),
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.4, intensity=0.85, position=1),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)
        selected = [b for b in beats if b.selected_for_generation]
        assert len(selected) == 1
        assert selected[0].event_type == EventType.LEAD_BATTLE

    def test_repeated_same_storyline_no_change(self):
        """Same battle narrated, then repeated without meaningful change → suppressed."""
        config = SchedulerConfig()
        config.storyline.cooldown_frames = 1
        config.storyline.battle_advance_threshold_sec = 0.5
        scheduler = EditorialScheduler(config)

        battle = lambda f, g: _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=g, intensity=0.8, position=1, frame=f)

        # Frame 0: narrate.
        beats0 = scheduler.schedule([battle(0, 1.0)], frame_id=0)
        assert beats0[0].selected_for_generation

        # Frame 2: gap barely changed (1.0 → 0.9, delta=0.1 < 0.5).
        beats2 = scheduler.schedule([battle(2, 0.9)], frame_id=2)
        assert not beats2[0].selected_for_generation
        assert beats2[0].suppression_reason == SuppressionReason.STORYLINE_NOT_ADVANCED

    def test_system_under_high_load(self):
        """Under high backpressure, non-critical events are dropped, safety survives."""
        config = SchedulerConfig()
        scheduler = EditorialScheduler(config)
        scheduler.queue_depth = config.backpressure.queue_hard_limit + 3

        candidates = [
            _evt(EventType.MIDFIELD_BATTLE, ["SAI", "NOR"], gap=0.8, position=6),
            _evt(EventType.RACE_CONTROL, safety=True, desc="RED FLAG"),
        ]
        beats = scheduler.schedule(candidates, frame_id=0)

        midfield = [b for b in beats if b.event_type == EventType.MIDFIELD_BATTLE]
        safety = [b for b in beats if b.event_type == EventType.RACE_CONTROL]

        assert not midfield[0].selected_for_generation
        assert midfield[0].suppression_reason == SuppressionReason.BACKPRESSURE_DROP
        assert safety[0].selected_for_generation


# ── 7. Schema validity ───────────────────────────────────────────────────

class TestSchemaValidity:
    def test_scheduled_beat_round_trip(self):
        beat = ScheduledBeat(
            beat_id="test_001",
            timestamp=TS,
            event_type=EventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
            storyline_id="battle_HAM_VER",
            priority_score=0.87,
            score_breakdown=ScoreBreakdown(
                contextual_importance=0.9,
                urgency=0.85,
                race_phase=0.3,
                storyline_relevance=0.8,
                confidence=1.0,
                backpressure_penalty=0.0,
            ),
            selected_for_generation=True,
            suppression_reason=SuppressionReason.NONE,
            retrieval_allowed=True,
            output_length_budget=150,
            source_frame_id=5,
            description="VER and HAM battling for the lead",
        )
        json_str = beat.model_dump_json()
        restored = ScheduledBeat.model_validate_json(json_str)
        assert restored.beat_id == "test_001"
        assert restored.score_breakdown.urgency == 0.85
        assert restored.suppression_reason == SuppressionReason.NONE

    def test_candidate_event_round_trip(self):
        evt = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5, intensity=0.9, position=1)
        json_str = evt.model_dump_json()
        restored = CandidateEvent.model_validate_json(json_str)
        assert restored.event_type == EventType.LEAD_BATTLE
        assert restored.gap_sec == 0.5

    def test_score_breakdown_defaults(self):
        sb = ScoreBreakdown()
        assert sb.contextual_importance == 0.0
        assert sb.backpressure_penalty == 0.0

    def test_suppression_reason_values(self):
        """All suppression reasons should be serializable."""
        for reason in SuppressionReason:
            beat = ScheduledBeat(
                beat_id="test",
                timestamp=TS,
                event_type=EventType.TELEMETRY_CHANGE,
                suppression_reason=reason,
            )
            data = json.loads(beat.model_dump_json())
            assert data["suppression_reason"] == reason.value


# ── 8. Snapshot-to-candidate adapter ─────────────────────────────────────

class TestSnapshotToCandidates:
    def test_battle_extracted_as_lead(self):
        snap = _snap(
            frame_id=0,
            battles=[
                BattleState(
                    driver_ahead="VER",
                    driver_behind="HAM",
                    gap_sec=0.5,
                    battle_intensity=0.85,
                )
            ],
        )
        candidates = extract_candidates(snap)
        lead_battles = [c for c in candidates if c.event_type == EventType.LEAD_BATTLE]
        assert len(lead_battles) == 1
        assert "VER" in lead_battles[0].involved_drivers

    def test_safety_car_transition_detected(self):
        prev = _snap(frame_id=0, track_status="GREEN FLAG", safety_car=False)
        curr = _snap(frame_id=1, track_status="SAFETY CAR", safety_car=True)
        candidates = extract_candidates(curr, prev)
        rc = [c for c in candidates if c.event_type == EventType.RACE_CONTROL]
        assert len(rc) == 1
        assert rc[0].safety_critical is True

    def test_no_rc_event_when_status_unchanged(self):
        prev = _snap(frame_id=0, track_status="GREEN FLAG")
        curr = _snap(frame_id=1, track_status="GREEN FLAG")
        candidates = extract_candidates(curr, prev)
        rc = [c for c in candidates if c.event_type == EventType.RACE_CONTROL]
        assert len(rc) == 0

    def test_pit_stop_detected(self):
        prev_drivers = {
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1, tire_compound="SOFT"),
        }
        curr_drivers = {
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1, tire_compound="HARD"),
        }
        prev = _snap(frame_id=0, drivers=prev_drivers)
        curr = _snap(frame_id=1, drivers=curr_drivers)
        candidates = extract_candidates(curr, prev)
        pits = [c for c in candidates if c.event_type == EventType.PIT_STRATEGY]
        assert len(pits) == 1
        assert "VER" in pits[0].involved_drivers

    def test_overtake_detected(self):
        prev_drivers = {
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1),
            "HAM": DriverRaceState(driver_number="44", abbreviation="HAM", position=2),
        }
        curr_drivers = {
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=2),
            "HAM": DriverRaceState(driver_number="44", abbreviation="HAM", position=1),
        }
        prev = _snap(frame_id=0, drivers=prev_drivers)
        curr = _snap(frame_id=1, drivers=curr_drivers)
        candidates = extract_candidates(curr, prev)
        overtakes = [c for c in candidates if c.event_type == EventType.OVERTAKE]
        assert len(overtakes) == 1
        assert overtakes[0].involved_drivers == ["HAM"]

    def test_empty_snapshot_yields_filler(self):
        """When nothing changed between two identical snapshots, a filler is emitted."""
        prev = _snap(frame_id=0, battles=[], drivers={
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1),
        })
        curr = _snap(frame_id=1, battles=[], drivers={
            "VER": DriverRaceState(driver_number="1", abbreviation="VER", position=1),
        })
        candidates = extract_candidates(curr, prev)
        assert len(candidates) >= 1
        assert any(c.event_type == EventType.TELEMETRY_CHANGE for c in candidates)


# ── 9. Observability counters ────────────────────────────────────────────

class TestObservabilityCounters:
    def test_counters_increment(self):
        scheduler = EditorialScheduler()
        candidates = [
            _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.3, intensity=0.9, position=1),
            _evt(EventType.TELEMETRY_CHANGE, confidence=0.3),
        ]
        scheduler.schedule(candidates, frame_id=0)

        c = scheduler.counters
        assert c.events_received == 2
        assert c.beats_selected >= 1
        assert c.beats_suppressed >= 1

    def test_queue_depth_high_water_mark(self):
        scheduler = EditorialScheduler()
        scheduler.queue_depth = 7
        scheduler.queue_depth = 3
        assert scheduler.counters.queue_depth_high_water == 7

    def test_counters_as_dict(self):
        c = SchedulerCounters()
        d = c.as_dict()
        assert "events_received" in d
        assert "beats_dropped" in d


# ── 10. Scoring helpers unit tests ────────────────────────────────────────

class TestScoringHelpers:
    def test_contextual_importance_tiers(self):
        config = SchedulerConfig()
        rc = score_contextual_importance(_evt(EventType.RACE_CONTROL), config)
        lb = score_contextual_importance(_evt(EventType.LEAD_BATTLE), config)
        tc = score_contextual_importance(_evt(EventType.TELEMETRY_CHANGE), config)
        assert rc > lb > tc

    def test_urgency_safety_maximal(self):
        assert score_urgency(_evt(EventType.RACE_CONTROL, safety=True)) == 1.0

    def test_urgency_scales_with_gap(self):
        close = score_urgency(_evt(gap=0.3))
        far = score_urgency(_evt(gap=1.8))
        assert close > far

    def test_race_phase_opening_laps(self):
        assert score_race_phase(lap_number=1, total_laps=57, safety_car_active=False) == 0.9

    def test_race_phase_closing_laps(self):
        assert score_race_phase(lap_number=55, total_laps=57, safety_car_active=False) == 0.8

    def test_race_phase_safety_car(self):
        assert score_race_phase(lap_number=30, total_laps=57, safety_car_active=True) == 0.7

    def test_race_phase_mid_race(self):
        assert score_race_phase(lap_number=30, total_laps=57, safety_car_active=False) == 0.3

    def test_storyline_relevance_new_storyline(self):
        config = SchedulerConfig()
        tracker = StorylineTracker(config)
        evt = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"])
        score = score_storyline_relevance(tracker, "battle_HAM_VER", evt, current_frame=0)
        assert score == 0.6  # novelty

    def test_storyline_relevance_continuation(self):
        config = SchedulerConfig()
        config.storyline.cooldown_frames = 1
        config.storyline.battle_advance_threshold_sec = 0.1
        tracker = StorylineTracker(config)
        tracker.record_narration("battle_HAM_VER", frame_id=0, gap_sec=1.0)

        evt = _evt(EventType.LEAD_BATTLE, ["VER", "HAM"], gap=0.5)  # advanced
        score = score_storyline_relevance(tracker, "battle_HAM_VER", evt, current_frame=3)
        assert score == 0.8  # continuation
