"""Tests for the even/odd schema bridge adapters."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure odd-phase package is importable (``code/`` contains ``src/`` which
# is a Python package with ``__init__.py``).
_CODE_ROOT = str(Path(__file__).resolve().parents[1] / "code")
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

# --- Even-phase imports ---
from f1_commentary.schemas.events import (
    CandidateEvent as EvenCandidateEvent,
    CandidateEventType,
    EventType as EvenEventType,
    GuardedCommentary as EvenGuardedCommentary,
)

# --- Odd-phase imports ---
from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState as OddTelemetryDriverState,
    GlobalState as OddGlobalState,
    SessionInfo as OddSessionInfo,
    TelemetryFrame as OddTelemetryFrame,
)
from src.telemetry_to_narrative.schemas.race_state import (
    BattleState as OddBattleState,
    DriverRaceState as OddDriverRaceState,
    GlobalRaceStateSummary as OddGlobalRaceStateSummary,
    RaceStateSnapshot as OddRaceStateSnapshot,
)
from src.telemetry_to_narrative.schemas.candidate_event import (
    CandidateEvent as OddCandidateEvent,
    EventType as OddEventType,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScheduledBeat as OddScheduledBeat,
)
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary as OddGeneratedCommentary,
)
from src.telemetry_to_narrative.schemas.final_commentary import (
    ConfidenceBand as OddConfidenceBand,
    FinalCommentary as OddFinalCommentary,
    RaceStateSummary as OddRaceStateSummary,
)

# --- Bridge imports ---
from f1_commentary.adapters.bridge import (
    EVEN_TO_ODD_EVENT_TYPE,
    ODD_TO_EVEN_EVENT_TYPE,
    even_candidate_to_odd,
    even_guarded_to_odd_generated,
    odd_beat_to_even_candidate,
    odd_commentary_to_even,
    odd_final_to_even,
    odd_snapshot_to_even,
    odd_telemetry_to_even,
)

NOW = datetime(2024, 9, 1, 14, 30, 0, tzinfo=timezone.utc)
SESSION_INFO = OddSessionInfo(year=2024, grand_prix="Monza", session_type="R")


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def odd_telemetry_frame() -> OddTelemetryFrame:
    return OddTelemetryFrame(
        frame_id=42,
        timestamp=NOW,
        session_time=timedelta(seconds=3661),
        session_info=SESSION_INFO,
        drivers=[
            OddTelemetryDriverState(
                driver_number="44",
                abbreviation="HAM",
                position=1,
                lap_number=15,
                gap_ahead_sec=None,
                tire_compound="MEDIUM",
                tire_age_laps=8,
                drs_active=False,
                speed_kph=320.5,
            ),
            OddTelemetryDriverState(
                driver_number="1",
                abbreviation="VER",
                position=2,
                lap_number=15,
                gap_ahead_sec=1.2,
                tire_compound="HARD",
                tire_age_laps=3,
                drs_active=True,
                speed_kph=322.1,
            ),
        ],
        global_state=OddGlobalState(
            track_status="Green",
            safety_car=False,
            air_temp_celsius=28.0,
            rainfall=False,
            session_clock="01:01:01",
        ),
    )


@pytest.fixture
def odd_race_state() -> OddRaceStateSnapshot:
    ham = OddDriverRaceState(
        driver_number="44",
        abbreviation="HAM",
        position=1,
        lap_number=20,
        gap_ahead_sec=None,
        tire_compound="MEDIUM",
        tire_age_laps=12,
        drs_active=False,
        speed_kph=310.0,
        relative_pace_delta=-0.3,
        catch_time_laps=None,
        degradation_score=0.02,
    )
    ver = OddDriverRaceState(
        driver_number="1",
        abbreviation="VER",
        position=2,
        lap_number=20,
        gap_ahead_sec=0.8,
        tire_compound="HARD",
        tire_age_laps=5,
        drs_active=True,
        speed_kph=315.0,
        relative_pace_delta=-0.5,
        catch_time_laps=1.6,
        degradation_score=0.01,
    )
    return OddRaceStateSnapshot(
        frame_id=100,
        timestamp=NOW,
        session_time=timedelta(seconds=4000),
        session_info=SESSION_INFO,
        global_state=OddGlobalRaceStateSummary(
            track_status="Green",
            safety_car=False,
            air_temp_celsius=29.0,
            rainfall=True,
        ),
        drivers={"HAM": ham, "VER": ver},
        active_battles=[
            OddBattleState(
                driver_ahead="HAM",
                driver_behind="VER",
                gap_sec=0.8,
            )
        ],
    )


@pytest.fixture
def even_candidate() -> EvenCandidateEvent:
    return EvenCandidateEvent(
        timestamp=NOW,
        candidate_type=CandidateEventType.OVERTAKE_OPPORTUNITY,
        involved_drivers=["VER", "HAM"],
        confidence=0.85,
        severity=0.6,
        explanation="VER closing on HAM with DRS",
        lap=20,
        session_key="2024_Monza_R",
    )


@pytest.fixture
def odd_scheduled_beat() -> OddScheduledBeat:
    return OddScheduledBeat(
        beat_id="beat-001",
        timestamp=NOW,
        event_type=OddEventType.LEAD_BATTLE,
        involved_drivers=["VER", "HAM"],
        storyline_id="battle_VER_HAM",
        priority_score=0.9,
        description="Intense battle for the lead",
        source_frame_id=100,
    )


@pytest.fixture
def odd_generated_commentary() -> OddGeneratedCommentary:
    return OddGeneratedCommentary(
        generation_id="gen-001",
        timestamp=NOW,
        beat_id="beat-001",
        storyline_id="battle_VER_HAM",
        event_type=OddEventType.LEAD_BATTLE,
        involved_drivers=["VER", "HAM"],
        model_variant="mistral-7b",
        prompt_text="Describe the battle between VER and HAM.",
        raw_output_text="Verstappen is all over Hamilton!  ",
        commentary_text="Verstappen is all over Hamilton!",
        latency_ms=245.0,
    )


@pytest.fixture
def odd_final_commentary() -> OddFinalCommentary:
    return OddFinalCommentary(
        final_id="final-001",
        timestamp=NOW,
        beat_id="beat-001",
        storyline_id="battle_VER_HAM",
        event_type=OddEventType.LEAD_BATTLE,
        involved_drivers=["VER", "HAM"],
        final_text="Verstappen closes in on Hamilton for the lead!",
        confidence_band=OddConfidenceBand.HIGH,
        replay_lag_seconds=2.5,
        race_state_summary=OddRaceStateSummary(
            lap=20, leader="HAM", flag_state="Green"
        ),
    )


# =========================================================================
# 1. odd_telemetry_to_even
# =========================================================================


class TestOddTelemetryToEven:
    def test_converts_drivers(self, odd_telemetry_frame: OddTelemetryFrame):
        result = odd_telemetry_to_even(odd_telemetry_frame)
        assert len(result.drivers) == 2
        ham = result.drivers[0]
        assert ham.driver_id == "HAM"
        assert ham.position == 1
        assert ham.tire_age == 8
        assert ham.drs_available is False
        assert ham.speed == 320.5

    def test_session_key(self, odd_telemetry_frame: OddTelemetryFrame):
        result = odd_telemetry_to_even(odd_telemetry_frame)
        assert result.session_key == "2024_Monza_R"

    def test_flag_and_weather(self, odd_telemetry_frame: OddTelemetryFrame):
        result = odd_telemetry_to_even(odd_telemetry_frame)
        assert result.flag_status == "Green"
        assert result.weather == "28.0°C"

    def test_session_clock(self, odd_telemetry_frame: OddTelemetryFrame):
        result = odd_telemetry_to_even(odd_telemetry_frame)
        assert result.session_clock == 3661.0

    def test_has_base_event_fields(self, odd_telemetry_frame: OddTelemetryFrame):
        result = odd_telemetry_to_even(odd_telemetry_frame)
        assert result.event_id is not None
        assert result.timestamp == NOW
        assert result.event_type == EvenEventType.TELEMETRY_FRAME


# =========================================================================
# 2. odd_snapshot_to_even
# =========================================================================


class TestOddSnapshotToEven:
    def test_drivers_dict_to_list(self, odd_race_state: OddRaceStateSnapshot):
        result = odd_snapshot_to_even(odd_race_state)
        assert isinstance(result.drivers, list)
        assert len(result.drivers) == 2

    def test_field_mapping(self, odd_race_state: OddRaceStateSnapshot):
        result = odd_snapshot_to_even(odd_race_state)
        ver = [d for d in result.drivers if d.driver_id == "VER"][0]
        assert ver.gap_ahead == 0.8
        assert ver.tire_age == 5
        assert ver.drs_available is True

    def test_derived_features(self, odd_race_state: OddRaceStateSnapshot):
        result = odd_snapshot_to_even(odd_race_state)
        assert "VER" in result.pace_deltas
        assert result.pace_deltas["VER"] == -0.5
        assert result.catch_times["VER"] == 1.6
        assert result.stint_deg_scores["VER"] == 0.01
        assert result.stint_deg_scores["HAM"] == 0.02

    def test_battle_pairs(self, odd_race_state: OddRaceStateSnapshot):
        result = odd_snapshot_to_even(odd_race_state)
        assert ("HAM", "VER") in result.battle_pairs

    def test_flag_weather(self, odd_race_state: OddRaceStateSnapshot):
        result = odd_snapshot_to_even(odd_race_state)
        assert result.flag_status == "Green"
        assert "Rain" in (result.weather or "")


# =========================================================================
# 3. even_candidate_to_odd
# =========================================================================


class TestEvenCandidateToOdd:
    def test_overtake_mapping(self, even_candidate: EvenCandidateEvent):
        result = even_candidate_to_odd(even_candidate)
        assert result.event_type == OddEventType.OVERTAKE

    def test_active_battle_low_severity(self):
        ev = EvenCandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.ACTIVE_BATTLE,
            involved_drivers=["NOR", "PIA"],
            confidence=0.7,
            severity=0.5,
            explanation="Midfield scrap",
        )
        result = even_candidate_to_odd(ev)
        assert result.event_type == OddEventType.MIDFIELD_BATTLE
        assert result.safety_critical is False

    def test_active_battle_high_severity(self):
        ev = EvenCandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.ACTIVE_BATTLE,
            involved_drivers=["VER", "HAM"],
            confidence=0.9,
            severity=0.85,
            explanation="Leaders fighting",
        )
        result = even_candidate_to_odd(ev)
        assert result.event_type == OddEventType.LEAD_BATTLE
        assert result.safety_critical is True

    def test_safety_car_mapping(self):
        ev = EvenCandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.SAFETY_CAR,
            involved_drivers=[],
            confidence=1.0,
            severity=1.0,
            explanation="Safety car deployed",
        )
        result = even_candidate_to_odd(ev)
        assert result.event_type == OddEventType.RACE_CONTROL
        assert result.safety_critical is True

    def test_all_event_types_mapped(self):
        """Every CandidateEventType has a mapping in the table."""
        for ct in CandidateEventType:
            assert ct in EVEN_TO_ODD_EVENT_TYPE

    def test_preserves_drivers_and_confidence(self, even_candidate: EvenCandidateEvent):
        result = even_candidate_to_odd(even_candidate)
        assert result.involved_drivers == ["VER", "HAM"]
        assert result.confidence == 0.85

    def test_explanation_to_description(self, even_candidate: EvenCandidateEvent):
        result = even_candidate_to_odd(even_candidate)
        assert result.description == "VER closing on HAM with DRS"


# =========================================================================
# 4. odd_beat_to_even_candidate
# =========================================================================


class TestOddBeatToEvenCandidate:
    def test_basic_conversion(self, odd_scheduled_beat: OddScheduledBeat):
        result = odd_beat_to_even_candidate(odd_scheduled_beat)
        assert result.candidate_type == CandidateEventType.LEAD_BATTLE
        assert result.involved_drivers == ["VER", "HAM"]
        assert result.confidence == 0.9
        assert result.explanation == "Intense battle for the lead"
        assert result.storyline_id == "battle_VER_HAM"

    def test_priority_clamped(self):
        beat = OddScheduledBeat(
            beat_id="b",
            timestamp=NOW,
            event_type=OddEventType.OVERTAKE,
            priority_score=1.5,  # above 1.0
        )
        result = odd_beat_to_even_candidate(beat)
        assert result.confidence == 1.0


# =========================================================================
# 5. odd_commentary_to_even
# =========================================================================


class TestOddCommentaryToEven:
    def test_field_mapping(self, odd_generated_commentary: OddGeneratedCommentary):
        result = odd_commentary_to_even(odd_generated_commentary)
        assert result.source_beat_id == "beat-001"
        assert result.generated_text == "Verstappen is all over Hamilton!"
        assert result.generation_latency_ms == 245.0
        assert result.model_variant == "mistral-7b"
        assert result.involved_drivers == ["VER", "HAM"]

    def test_uses_processed_text(self):
        gen = OddGeneratedCommentary(
            generation_id="g2",
            timestamp=NOW,
            beat_id="b2",
            event_type=OddEventType.OVERTAKE,
            model_variant="mock",
            prompt_text="p",
            raw_output_text="raw stuff here",
            commentary_text="",  # empty -> should fall back to raw
        )
        result = odd_commentary_to_even(gen)
        assert result.generated_text == "raw stuff here"


# =========================================================================
# 6. even_guarded_to_odd_generated
# =========================================================================


class TestEvenGuardedToOddGenerated:
    def test_preserves_adjusted_text(
        self, odd_generated_commentary: OddGeneratedCommentary
    ):
        guarded = EvenGuardedCommentary(
            timestamp=NOW,
            source_commentary_id="c1",
            original_text="Verstappen is all over Hamilton!",
            adjusted_text="Verstappen is closing rapidly on Hamilton!",
        )
        result = even_guarded_to_odd_generated(guarded, odd_generated_commentary)
        assert result.commentary_text == "Verstappen is closing rapidly on Hamilton!"
        # All other fields from original are preserved
        assert result.generation_id == "gen-001"
        assert result.beat_id == "beat-001"
        assert result.latency_ms == 245.0
        assert result.raw_output_text == "Verstappen is all over Hamilton!  "


# =========================================================================
# 7. odd_final_to_even
# =========================================================================


class TestOddFinalToEven:
    def test_field_mapping(self, odd_final_commentary: OddFinalCommentary):
        result = odd_final_to_even(odd_final_commentary)
        assert result.final_text == "Verstappen closes in on Hamilton for the lead!"
        assert result.storyline_id == "battle_VER_HAM"
        assert result.involved_drivers == ["VER", "HAM"]

    def test_confidence_band_string(self, odd_final_commentary: OddFinalCommentary):
        result = odd_final_to_even(odd_final_commentary)
        assert result.confidence_band == "high"

    def test_lag_conversion(self, odd_final_commentary: OddFinalCommentary):
        result = odd_final_to_even(odd_final_commentary)
        assert result.replay_lag_ms == 2500.0

    def test_race_state_summary_dict(self, odd_final_commentary: OddFinalCommentary):
        result = odd_final_to_even(odd_final_commentary)
        assert isinstance(result.race_state_summary, dict)
        assert result.race_state_summary["lap"] == 20
        assert result.race_state_summary["leader"] == "HAM"

    def test_base_event_wrapper(self, odd_final_commentary: OddFinalCommentary):
        result = odd_final_to_even(odd_final_commentary)
        assert result.event_id is not None
        assert result.timestamp == NOW
        assert result.event_type == EvenEventType.FINAL_COMMENTARY


# =========================================================================
# 8. Round-trip tests
# =========================================================================


class TestRoundTrip:
    def test_candidate_event_round_trip(self):
        """even CandidateEvent -> odd CandidateEvent -> even CandidateEvent
        preserves essential fields."""
        original = EvenCandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
            confidence=0.9,
            severity=0.7,
            explanation="Fight for the lead",
            lap=30,
            session_key="2024_Monza_R",
        )

        odd = even_candidate_to_odd(original)
        assert odd.event_type == OddEventType.LEAD_BATTLE

        # Build a ScheduledBeat-like object from the odd CandidateEvent
        # to test the reverse path via odd_beat_to_even_candidate
        beat = OddScheduledBeat(
            beat_id=odd.event_id,
            timestamp=odd.timestamp,
            event_type=odd.event_type,
            involved_drivers=odd.involved_drivers,
            priority_score=odd.confidence,
            description=odd.description,
        )
        back = odd_beat_to_even_candidate(beat)

        assert back.candidate_type == CandidateEventType.LEAD_BATTLE
        assert back.involved_drivers == original.involved_drivers
        assert back.confidence == original.confidence
        assert back.explanation == original.explanation

    def test_commentary_round_trip(
        self, odd_generated_commentary: OddGeneratedCommentary
    ):
        """odd GeneratedCommentary -> even -> guarded -> back to odd
        preserves the adjusted text."""
        even = odd_commentary_to_even(odd_generated_commentary)
        assert even.generated_text == "Verstappen is all over Hamilton!"

        guarded = EvenGuardedCommentary(
            timestamp=NOW,
            source_commentary_id=even.event_id,
            original_text=even.generated_text,
            adjusted_text="Verstappen puts Hamilton under serious pressure!",
        )
        back = even_guarded_to_odd_generated(guarded, odd_generated_commentary)
        assert (
            back.commentary_text
            == "Verstappen puts Hamilton under serious pressure!"
        )
        assert back.beat_id == odd_generated_commentary.beat_id
