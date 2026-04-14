"""Safety car, yellow flag, and red flag recognizer."""

from datetime import datetime, timezone

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    RaceStateSnapshot,
)

# Mapping from flag_status strings to event type and severity
_FLAG_MAP: dict[str, tuple[CandidateEventType, float, float]] = {
    "SC": (CandidateEventType.SAFETY_CAR, 0.97, 0.9),
    "VSC": (CandidateEventType.SAFETY_CAR, 0.95, 0.7),
    "YELLOW": (CandidateEventType.YELLOW_FLAG, 0.95, 0.6),
    "RED": (CandidateEventType.RED_FLAG, 0.98, 1.0),
}


class SafetyCarRecognizer(EventRecognizer):
    """Detect safety car, VSC, yellow, and red flag conditions."""

    @property
    def name(self) -> str:
        return "SafetyCarRecognizer"

    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        flag = snapshot.flag_status
        if not flag or flag.upper() == "GREEN":
            return []

        flag_upper = flag.upper()
        mapping = _FLAG_MAP.get(flag_upper)
        if mapping is None:
            return []

        candidate_type, confidence, severity = mapping

        # All drivers are involved in flag conditions
        involved = [d.driver_id for d in snapshot.drivers]

        explanation_map = {
            "SC": "Safety Car deployed",
            "VSC": "Virtual Safety Car deployed",
            "YELLOW": "Yellow flag shown",
            "RED": "Red flag shown - session stopped",
        }

        return [
            CandidateEvent(
                timestamp=datetime.now(tz=timezone.utc),
                candidate_type=candidate_type,
                involved_drivers=involved,
                confidence=confidence,
                severity=severity,
                explanation=explanation_map.get(flag_upper, f"Flag: {flag_upper}"),
                triggering_features={"flag_status": flag_upper},
                lap=snapshot.lap,
                session_key=snapshot.session_key,
            )
        ]
