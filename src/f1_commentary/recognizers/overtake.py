"""Overtake and DRS threat recognizer."""

from datetime import datetime, timezone

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    RaceStateSnapshot,
)


class OvertakeRecognizer(EventRecognizer):
    """Detect overtake opportunities and DRS threats from gap/pace data."""

    @property
    def name(self) -> str:
        return "OvertakeRecognizer"

    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []

        for driver in snapshot.drivers:
            driver_id = driver.driver_id
            gap_ahead = driver.gap_ahead
            if gap_ahead is None or gap_ahead <= 0:
                continue

            pace_delta = snapshot.pace_deltas.get(driver_id, 0.0)

            # Overtake opportunity: very close and closing
            if gap_ahead < 0.5 and pace_delta < -0.2:
                confidence = min(1.0, 0.6 + (0.5 - gap_ahead) + abs(pace_delta) * 0.5)
                confidence = round(min(confidence, 1.0), 3)

                # Find the driver ahead
                driver_ahead = self._find_driver_ahead(snapshot, driver.position)
                involved = [driver_id]
                if driver_ahead:
                    involved.append(driver_ahead)

                events.append(
                    CandidateEvent(
                        timestamp=datetime.now(tz=timezone.utc),
                        candidate_type=CandidateEventType.OVERTAKE_OPPORTUNITY,
                        involved_drivers=involved,
                        confidence=confidence,
                        severity=0.7,
                        explanation=(
                            f"{driver_id} is {gap_ahead:.2f}s behind "
                            f"{driver_ahead or 'leader'} and closing at "
                            f"{pace_delta:+.2f}s/lap"
                        ),
                        triggering_features={
                            "gap_ahead": gap_ahead,
                            "pace_delta": pace_delta,
                            "drs_available": driver.drs_available,
                        },
                        lap=snapshot.lap,
                        session_key=snapshot.session_key,
                    )
                )
            # DRS threat: within DRS range with DRS available
            elif gap_ahead < 1.0 and driver.drs_available:
                confidence = round(min(1.0, 0.5 + (1.0 - gap_ahead) * 0.5), 3)

                driver_ahead = self._find_driver_ahead(snapshot, driver.position)
                involved = [driver_id]
                if driver_ahead:
                    involved.append(driver_ahead)

                events.append(
                    CandidateEvent(
                        timestamp=datetime.now(tz=timezone.utc),
                        candidate_type=CandidateEventType.DRS_THREAT,
                        involved_drivers=involved,
                        confidence=confidence,
                        severity=0.5,
                        explanation=(
                            f"{driver_id} has DRS available, {gap_ahead:.2f}s "
                            f"behind {driver_ahead or 'leader'}"
                        ),
                        triggering_features={
                            "gap_ahead": gap_ahead,
                            "drs_available": True,
                            "pace_delta": pace_delta,
                        },
                        lap=snapshot.lap,
                        session_key=snapshot.session_key,
                    )
                )

        return events

    @staticmethod
    def _find_driver_ahead(
        snapshot: RaceStateSnapshot, position: int
    ) -> str | None:
        """Find the driver in the position directly ahead."""
        for d in snapshot.drivers:
            if d.position == position - 1:
                return d.driver_id
        return None
