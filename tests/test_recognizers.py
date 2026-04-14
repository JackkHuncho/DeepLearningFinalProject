"""Tests for F1 event recognizers (Phase 4)."""

from datetime import datetime, timezone

import pytest

from f1_commentary.schemas.events import (
    CandidateEventType,
    DriverState,
    EventType,
    RaceStateSnapshot,
)
from f1_commentary.recognizers.overtake import OvertakeRecognizer
from f1_commentary.recognizers.battle import BattleRecognizer
from f1_commentary.recognizers.pit_strategy import PitStrategyRecognizer
from f1_commentary.recognizers.tire_deg import TireDegRecognizer
from f1_commentary.recognizers.safety_car import SafetyCarRecognizer
from f1_commentary.recognizers.engine import RecognitionEngine


def _make_snapshot(**kwargs) -> RaceStateSnapshot:
    """Helper to build a RaceStateSnapshot with sensible defaults."""
    defaults = dict(
        timestamp=datetime.now(tz=timezone.utc),
        event_type=EventType.RACE_STATE_UPDATE,
        session_key="2024_Monza_R",
        lap=25,
        drivers=[],
        gaps={},
        pace_deltas={},
        catch_times={},
        stint_deg_scores={},
        battle_pairs=[],
        flag_status="GREEN",
    )
    defaults.update(kwargs)
    return RaceStateSnapshot(**defaults)


def _make_driver(**kwargs) -> DriverState:
    """Helper to build a DriverState with sensible defaults."""
    defaults = dict(
        driver_id="HAM",
        position=1,
        lap_number=25,
        gap_ahead=None,
        tire_compound="MEDIUM",
        tire_age=10,
        drs_available=False,
        speed=320.0,
        last_lap_time=81.5,
    )
    defaults.update(kwargs)
    return DriverState(**defaults)


# -------------------------------------------------------------------------
# OvertakeRecognizer
# -------------------------------------------------------------------------


class TestOvertakeRecognizer:
    def test_drs_threat(self):
        """HAM 0.8s behind VER, DRS available -> DRS_THREAT."""
        recognizer = OvertakeRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1, gap_ahead=None, drs_available=False),
                _make_driver(driver_id="HAM", position=2, gap_ahead=0.8, drs_available=True),
            ],
            pace_deltas={"HAM": 0.0, "VER": 0.0},
        )
        events = recognizer.recognize(snapshot)
        drs_events = [e for e in events if e.candidate_type == CandidateEventType.DRS_THREAT]
        assert len(drs_events) == 1
        assert "HAM" in drs_events[0].involved_drivers
        assert "VER" in drs_events[0].involved_drivers

    def test_overtake_opportunity(self):
        """HAM 0.3s behind VER, pace_delta -0.3 -> OVERTAKE_OPPORTUNITY."""
        recognizer = OvertakeRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1, gap_ahead=None, drs_available=False),
                _make_driver(driver_id="HAM", position=2, gap_ahead=0.3, drs_available=False),
            ],
            pace_deltas={"HAM": -0.3, "VER": 0.0},
        )
        events = recognizer.recognize(snapshot)
        overtake_events = [
            e for e in events if e.candidate_type == CandidateEventType.OVERTAKE_OPPORTUNITY
        ]
        assert len(overtake_events) == 1
        assert "HAM" in overtake_events[0].involved_drivers
        assert "VER" in overtake_events[0].involved_drivers


# -------------------------------------------------------------------------
# BattleRecognizer
# -------------------------------------------------------------------------


class TestBattleRecognizer:
    def test_three_driver_battle(self):
        """3 drivers within 1.5s gaps -> detects battle."""
        recognizer = BattleRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1, gap_ahead=None),
                _make_driver(driver_id="HAM", position=2, gap_ahead=0.8),
                _make_driver(driver_id="NOR", position=3, gap_ahead=0.7),
            ],
        )
        events = recognizer.recognize(snapshot)
        # Should produce at least one battle event
        assert len(events) >= 1
        # Find a battle event that includes all three drivers
        battle_events = [
            e
            for e in events
            if e.candidate_type
            in (
                CandidateEventType.LEAD_BATTLE,
                CandidateEventType.MIDFIELD_BATTLE,
                CandidateEventType.ACTIVE_BATTLE,
            )
        ]
        assert len(battle_events) >= 1
        all_involved = set()
        for ev in battle_events:
            all_involved.update(ev.involved_drivers)
        assert {"VER", "HAM", "NOR"}.issubset(all_involved)

    def test_lead_battle_positions_1_3(self):
        """Positions 1-3 in battle -> LEAD_BATTLE."""
        recognizer = BattleRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1, gap_ahead=None),
                _make_driver(driver_id="HAM", position=2, gap_ahead=0.5),
                _make_driver(driver_id="NOR", position=3, gap_ahead=0.6),
            ],
        )
        events = recognizer.recognize(snapshot)
        lead_events = [
            e for e in events if e.candidate_type == CandidateEventType.LEAD_BATTLE
        ]
        assert len(lead_events) >= 1

    def test_midfield_battle_positions_5_7(self):
        """Positions 5-7 -> MIDFIELD_BATTLE."""
        recognizer = BattleRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="ALO", position=5, gap_ahead=None),
                _make_driver(driver_id="STR", position=6, gap_ahead=0.6),
                _make_driver(driver_id="GAS", position=7, gap_ahead=0.9),
            ],
        )
        events = recognizer.recognize(snapshot)
        midfield_events = [
            e for e in events if e.candidate_type == CandidateEventType.MIDFIELD_BATTLE
        ]
        assert len(midfield_events) >= 1


# -------------------------------------------------------------------------
# PitStrategyRecognizer
# -------------------------------------------------------------------------


class TestPitStrategyRecognizer:
    def test_pit_strategy_trigger_over_stint(self):
        """SOFT tires at 22 laps (typical 18) -> PIT_STRATEGY_TRIGGER."""
        recognizer = PitStrategyRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(
                    driver_id="HAM",
                    position=3,
                    tire_compound="SOFT",
                    tire_age=22,
                ),
            ],
            stint_deg_scores={"HAM": 0.3},
        )
        events = recognizer.recognize(snapshot)
        pit_trigger = [
            e for e in events if e.candidate_type == CandidateEventType.PIT_STRATEGY_TRIGGER
        ]
        assert len(pit_trigger) == 1
        assert "HAM" in pit_trigger[0].involved_drivers

    def test_pit_stop_high_deg_score(self):
        """deg_score 0.8 -> PIT_STOP."""
        recognizer = PitStrategyRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(
                    driver_id="VER",
                    position=1,
                    tire_compound="MEDIUM",
                    tire_age=10,
                ),
            ],
            stint_deg_scores={"VER": 0.8},
        )
        events = recognizer.recognize(snapshot)
        pit_stop = [
            e for e in events if e.candidate_type == CandidateEventType.PIT_STOP
        ]
        assert len(pit_stop) == 1
        assert "VER" in pit_stop[0].involved_drivers


# -------------------------------------------------------------------------
# TireDegRecognizer
# -------------------------------------------------------------------------


class TestTireDegRecognizer:
    def test_tire_degradation_above_threshold(self):
        """stint_deg_score 0.6 (> threshold 0.5) -> TIRE_DEGRADATION."""
        recognizer = TireDegRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="HAM", position=3, tire_compound="SOFT", tire_age=15),
            ],
            stint_deg_scores={"HAM": 0.6},
        )
        events = recognizer.recognize(snapshot)
        deg_events = [
            e for e in events if e.candidate_type == CandidateEventType.TIRE_DEGRADATION
        ]
        assert len(deg_events) == 1
        assert "HAM" in deg_events[0].involved_drivers


# -------------------------------------------------------------------------
# SafetyCarRecognizer
# -------------------------------------------------------------------------


class TestSafetyCarRecognizer:
    def test_safety_car(self):
        """flag_status='SC' -> SAFETY_CAR."""
        recognizer = SafetyCarRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1),
                _make_driver(driver_id="HAM", position=2, gap_ahead=1.5),
            ],
            flag_status="SC",
        )
        events = recognizer.recognize(snapshot)
        assert len(events) == 1
        assert events[0].candidate_type == CandidateEventType.SAFETY_CAR

    def test_red_flag(self):
        """flag_status='RED' -> RED_FLAG."""
        recognizer = SafetyCarRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1),
                _make_driver(driver_id="HAM", position=2, gap_ahead=1.5),
            ],
            flag_status="RED",
        )
        events = recognizer.recognize(snapshot)
        assert len(events) == 1
        assert events[0].candidate_type == CandidateEventType.RED_FLAG

    def test_green_flag_no_events(self):
        """flag_status='GREEN' -> no events."""
        recognizer = SafetyCarRecognizer()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1),
                _make_driver(driver_id="HAM", position=2, gap_ahead=1.5),
            ],
            flag_status="GREEN",
        )
        events = recognizer.recognize(snapshot)
        assert len(events) == 0


# -------------------------------------------------------------------------
# RecognitionEngine
# -------------------------------------------------------------------------


class TestRecognitionEngine:
    def test_multiple_triggers_sorted_by_confidence(self):
        """Snapshot with multiple triggers -> returns sorted by confidence desc."""
        engine = RecognitionEngine()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(driver_id="VER", position=1, gap_ahead=None, drs_available=False),
                _make_driver(
                    driver_id="HAM",
                    position=2,
                    gap_ahead=0.3,
                    drs_available=False,
                    tire_compound="SOFT",
                    tire_age=22,
                ),
                _make_driver(driver_id="NOR", position=3, gap_ahead=0.7, drs_available=True),
            ],
            pace_deltas={"HAM": -0.3, "VER": 0.0, "NOR": 0.0},
            stint_deg_scores={"HAM": 0.8},
            flag_status="GREEN",
        )
        candidates = engine.process(snapshot)
        assert len(candidates) >= 2
        # Verify sorted by confidence descending
        for i in range(len(candidates) - 1):
            assert candidates[i].confidence >= candidates[i + 1].confidence


# -------------------------------------------------------------------------
# No false positives
# -------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_clean_snapshot_no_events(self):
        """Clean snapshot: big gaps, fresh tires, green flag -> empty list."""
        engine = RecognitionEngine()
        snapshot = _make_snapshot(
            drivers=[
                _make_driver(
                    driver_id="VER",
                    position=1,
                    gap_ahead=None,
                    tire_compound="HARD",
                    tire_age=5,
                    drs_available=False,
                ),
                _make_driver(
                    driver_id="HAM",
                    position=2,
                    gap_ahead=5.0,
                    tire_compound="HARD",
                    tire_age=5,
                    drs_available=False,
                ),
                _make_driver(
                    driver_id="NOR",
                    position=3,
                    gap_ahead=4.5,
                    tire_compound="HARD",
                    tire_age=5,
                    drs_available=False,
                ),
            ],
            pace_deltas={"VER": 0.0, "HAM": 0.0, "NOR": 0.0},
            stint_deg_scores={"VER": 0.1, "HAM": 0.1, "NOR": 0.1},
            flag_status="GREEN",
        )
        candidates = engine.process(snapshot)
        assert candidates == []
