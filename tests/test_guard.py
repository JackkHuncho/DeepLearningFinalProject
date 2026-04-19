"""Tests for the grounding guard (Phase 10)."""

from datetime import datetime, timezone

import pytest

from f1_commentary.schemas.events import (
    ClaimSupport,
    DriverState,
    EventType,
    GeneratedCommentary,
    GuardedCommentary,
    RaceStateSnapshot,
    RetrievalResult,
)
from f1_commentary.config import GuardConfig
from f1_commentary.guard.grounding_guard import GroundingGuard


# ---------------------------------------------------------------------------
# Helpers to build synthetic test data
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_state(
    drivers: list[DriverState] | None = None,
    lap: int = 42,
    gaps: dict[str, float] | None = None,
    pace_deltas: dict[str, float] | None = None,
) -> RaceStateSnapshot:
    if drivers is None:
        drivers = [
            DriverState(driver_id="VER", position=1, lap_number=42, gap_ahead=None,
                        tire_compound="HARD", tire_age=20),
            DriverState(driver_id="NOR", position=2, lap_number=42, gap_ahead=1.2,
                        tire_compound="MEDIUM", tire_age=10),
            DriverState(driver_id="LEC", position=3, lap_number=42, gap_ahead=0.8,
                        tire_compound="SOFT", tire_age=5),
            DriverState(driver_id="HAM", position=4, lap_number=42, gap_ahead=2.1,
                        tire_compound="MEDIUM", tire_age=15),
            DriverState(driver_id="SAI", position=5, lap_number=42, gap_ahead=3.0,
                        tire_compound="HARD", tire_age=22),
        ]
    return RaceStateSnapshot(
        timestamp=_now(),
        session_key="2024_Monza_R",
        lap=lap,
        drivers=drivers,
        gaps=gaps or {},
        pace_deltas=pace_deltas or {},
    )


def _make_commentary(text: str, drivers: list[str] | None = None) -> GeneratedCommentary:
    return GeneratedCommentary(
        timestamp=_now(),
        source_beat_id="beat-001",
        prompt_text="",
        generated_text=text,
        involved_drivers=drivers or [],
    )


def _make_retrieval(chunks: list[dict] | None = None, gated_out: bool = False) -> RetrievalResult:
    return RetrievalResult(
        timestamp=_now(),
        query_event_id="q-001",
        tier=1,
        chunks=chunks or [],
        gated_out=gated_out,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSupportedFactualClaim:
    """Commentary says 'HAM P4' and state confirms position 4."""

    def test_position_observed_high_confidence(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        commentary = _make_commentary("HAM is running in P4 and looking strong.")

        result = guard.check(commentary, state)

        assert isinstance(result, GuardedCommentary)
        pos_claims = [c for c in result.claims if "P4" in c.claim]
        assert len(pos_claims) >= 1
        assert pos_claims[0].support_type == "observed"
        assert pos_claims[0].confidence >= 0.8
        assert result.confidence_band == "high"


class TestInferredClaim:
    """Commentary says 'closing the gap' and pace_deltas confirm negative delta."""

    def test_closing_gap_inferred(self) -> None:
        guard = GroundingGuard()
        state = _make_state(pace_deltas={"NOR": -0.3})
        commentary = _make_commentary("NOR is closing the gap to VER lap after lap.")

        result = guard.check(commentary, state)

        closing_claims = [c for c in result.claims if "closing" in c.claim.lower()]
        assert len(closing_claims) >= 1
        assert closing_claims[0].support_type == "inferred"
        assert closing_claims[0].confidence >= 0.5


class TestUnsupportedClaim:
    """Commentary says 'Hamilton will win' with no evidence — gets softened."""

    def test_will_win_unsupported_and_softened(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        commentary = _make_commentary("HAM will win the race from P4.")

        result = guard.check(commentary, state)

        win_claims = [c for c in result.claims if c.support_type == "unsupported"
                      and "will win" in c.claim.lower()]
        assert len(win_claims) >= 1
        # Text must be softened — the original assertive phrase should be replaced
        assert result.adjusted_text != result.original_text
        # The original text is preserved
        assert "will win" in result.original_text.lower()


class TestMixedClaims:
    """Some claims are supported, some are not → confidence_band 'medium'."""

    def test_mixed_gives_medium(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        # P4 is correct for HAM (observed), "will overtake" is unsupported
        commentary = _make_commentary(
            "HAM is in P4 and will overtake LEC on the next lap."
        )

        result = guard.check(commentary, state)

        types = {c.support_type for c in result.claims}
        assert "observed" in types or "inferred" in types
        assert "unsupported" in types
        assert result.confidence_band == "medium"


class TestAllSupported:
    """Every claim verifiable → confidence_band 'high'."""

    def test_all_observed_high(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        # VER P1 and NOR P2 are both correct
        commentary = _make_commentary("VER holds P1, NOR is in P2.")

        result = guard.check(commentary, state)

        assert all(
            c.support_type in ("observed", "inferred", "retrieved")
            for c in result.claims
        )
        assert result.confidence_band == "high"


class TestGapClaimVerification:
    """'0.8s gap' matches gap data in state."""

    def test_gap_matches_driver(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        # LEC gap_ahead = 0.8
        commentary = _make_commentary("LEC is just 0.8s behind NOR.")

        result = guard.check(commentary, state)

        gap_claims = [c for c in result.claims if "0.8" in c.claim]
        assert len(gap_claims) >= 1
        assert gap_claims[0].support_type == "observed"


class TestTireClaimVerification:
    """'medium tires, 15 laps old' matches HAM driver state."""

    def test_tire_compound_and_age(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        commentary = _make_commentary(
            "HAM on medium tires, 15 laps old, managing the pace."
        )

        result = guard.check(commentary, state)

        tire_claims = [c for c in result.claims if "medium" in c.claim.lower()]
        age_claims = [c for c in result.claims if "15 laps old" in c.claim]
        assert len(tire_claims) >= 1
        assert tire_claims[0].support_type == "observed"
        assert len(age_claims) >= 1
        assert age_claims[0].support_type == "observed"


class TestNoClaimsExtractable:
    """Generic commentary with no verifiable claims → 'medium', no modifications."""

    def test_generic_commentary(self) -> None:
        guard = GroundingGuard()
        state = _make_state()
        commentary = _make_commentary(
            "What a thrilling race we are witnessing today!"
        )

        result = guard.check(commentary, state)

        assert result.confidence_band == "medium"
        assert result.adjusted_text == result.original_text
