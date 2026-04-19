"""Tests for the EventBus."""

from datetime import datetime, timezone

from f1_commentary.bus.event_bus import EventBus
from f1_commentary.schemas.events import (
    CandidateEvent,
    CandidateEventType,
    EventType,
    FinalCommentary,
    TelemetryFrame,
)

NOW = datetime.now(tz=timezone.utc)


def _telemetry(lap: int = 1) -> TelemetryFrame:
    return TelemetryFrame(timestamp=NOW, session_key="s", lap=lap, drivers=[])


def _final(text: str = "Go!") -> FinalCommentary:
    return FinalCommentary(timestamp=NOW, final_text=text)


class TestEventBus:
    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(EventType.TELEMETRY_FRAME, received.append)
        event = _telemetry()
        bus.publish(event)
        assert received == [event]

    def test_type_specific_routing(self) -> None:
        """Subscriber for TELEMETRY_FRAME should NOT get FINAL_COMMENTARY."""
        bus = EventBus()
        tel_events = []
        bus.subscribe(EventType.TELEMETRY_FRAME, tel_events.append)
        bus.publish(_final("ignored"))
        assert tel_events == []

    def test_global_subscriber_receives_all(self) -> None:
        bus = EventBus()
        all_events = []
        bus.subscribe_all(all_events.append)
        bus.publish(_telemetry())
        bus.publish(_final())
        assert len(all_events) == 2

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        a, b = [], []
        bus.subscribe(EventType.TELEMETRY_FRAME, a.append)
        bus.subscribe(EventType.TELEMETRY_FRAME, b.append)
        event = _telemetry()
        bus.publish(event)
        assert a == [event]
        assert b == [event]

    def test_publish_no_subscribers(self) -> None:
        """Publishing with no subscribers should not raise."""
        bus = EventBus()
        bus.publish(_telemetry())  # should be fine

    def test_clear(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(EventType.TELEMETRY_FRAME, received.append)
        bus.subscribe_all(received.append)
        bus.clear()
        bus.publish(_telemetry())
        assert received == []

    def test_mixed_type_and_global(self) -> None:
        bus = EventBus()
        typed, glob = [], []
        bus.subscribe(EventType.CANDIDATE_EVENT, typed.append)
        bus.subscribe_all(glob.append)
        ce = CandidateEvent(
            timestamp=NOW,
            candidate_type=CandidateEventType.PIT_STOP,
            involved_drivers=["HAM"],
            confidence=0.7,
        )
        bus.publish(ce)
        assert typed == [ce]
        assert glob == [ce]
