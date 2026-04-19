"""Recognition engine that orchestrates all recognizers."""

from f1_commentary.recognizers.base import EventRecognizer
from f1_commentary.schemas.events import CandidateEvent, RaceStateSnapshot


class RecognitionEngine:
    """Run all recognizers on a snapshot and collect candidate events."""

    def __init__(self, recognizers: list[EventRecognizer] | None = None) -> None:
        if recognizers is None:
            from f1_commentary.recognizers import create_all_recognizers

            recognizers = create_all_recognizers()
        self.recognizers = recognizers

    def process(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        """Run all recognizers on a snapshot, return candidates sorted by confidence desc."""
        candidates: list[CandidateEvent] = []
        for recognizer in self.recognizers:
            candidates.extend(recognizer.recognize(snapshot))
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
