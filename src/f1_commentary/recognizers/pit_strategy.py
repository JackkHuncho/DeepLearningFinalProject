"""Pit strategy recognizer."""

from datetime import datetime, timezone

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    RaceStateSnapshot,
)

# Typical stint lengths by compound (laps)
STINT_LENGTHS: dict[str, int] = {
    "SOFT": 18,
    "MEDIUM": 25,
    "HARD": 35,
    "INTER": 30,
    "WET": 25,
}


class PitStrategyRecognizer(EventRecognizer):
    """Detect when a driver is likely to pit based on tire age and degradation."""

    @property
    def name(self) -> str:
        return "PitStrategyRecognizer"

    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []

        for driver in snapshot.drivers:
            driver_id = driver.driver_id
            tire_compound = driver.tire_compound
            tire_age = driver.tire_age
            deg_score = snapshot.stint_deg_scores.get(driver_id, 0.0)

            # Check tire age vs typical stint length
            if tire_compound and tire_age is not None:
                typical = STINT_LENGTHS.get(tire_compound.upper(), 25)
                if tire_age > typical:
                    overshoot = tire_age - typical
                    confidence = round(min(1.0, 0.6 + overshoot * 0.05), 3)
                    events.append(
                        CandidateEvent(
                            timestamp=datetime.now(tz=timezone.utc),
                            candidate_type=CandidateEventType.PIT_STRATEGY_TRIGGER,
                            involved_drivers=[driver_id],
                            confidence=confidence,
                            severity=0.6,
                            explanation=(
                                f"{driver_id} has {tire_compound} tires at "
                                f"{tire_age} laps (typical stint: {typical})"
                            ),
                            triggering_features={
                                "tire_compound": tire_compound,
                                "tire_age": tire_age,
                                "typical_stint": typical,
                                "overshoot_laps": overshoot,
                            },
                            lap=snapshot.lap,
                            session_key=snapshot.session_key,
                        )
                    )

            # Check for pace drop suggesting tire cliff
            if deg_score > 0.7:
                confidence = round(min(1.0, 0.5 + deg_score * 0.4), 3)
                events.append(
                    CandidateEvent(
                        timestamp=datetime.now(tz=timezone.utc),
                        candidate_type=CandidateEventType.PIT_STOP,
                        involved_drivers=[driver_id],
                        confidence=confidence,
                        severity=0.7,
                        explanation=(
                            f"{driver_id} showing significant pace drop "
                            f"(deg score: {deg_score:.2f}), likely needs pit stop"
                        ),
                        triggering_features={
                            "stint_deg_score": deg_score,
                            "tire_compound": tire_compound,
                            "tire_age": tire_age,
                        },
                        lap=snapshot.lap,
                        session_key=snapshot.session_key,
                    )
                )

        return events
