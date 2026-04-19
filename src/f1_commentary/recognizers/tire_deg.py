"""Tire degradation recognizer."""

from datetime import datetime, timezone

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    RaceStateSnapshot,
)


class TireDegRecognizer(EventRecognizer):
    """Detect significant tire degradation from stint deg scores."""

    DEG_THRESHOLD = 0.5

    @property
    def name(self) -> str:
        return "TireDegRecognizer"

    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []

        for driver_id, deg_score in snapshot.stint_deg_scores.items():
            if deg_score <= self.DEG_THRESHOLD:
                continue

            # Higher confidence with higher deg score
            confidence = round(min(1.0, 0.4 + deg_score * 0.6), 3)
            severity = round(min(1.0, deg_score), 3)

            # Find tire info for context
            tire_compound = None
            tire_age = None
            for d in snapshot.drivers:
                if d.driver_id == driver_id:
                    tire_compound = d.tire_compound
                    tire_age = d.tire_age
                    break

            events.append(
                CandidateEvent(
                    timestamp=datetime.now(tz=timezone.utc),
                    candidate_type=CandidateEventType.TIRE_DEGRADATION,
                    involved_drivers=[driver_id],
                    confidence=confidence,
                    severity=severity,
                    explanation=(
                        f"{driver_id} experiencing tire degradation "
                        f"(score: {deg_score:.2f})"
                        + (f" on {tire_compound} lap {tire_age}" if tire_compound else "")
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
