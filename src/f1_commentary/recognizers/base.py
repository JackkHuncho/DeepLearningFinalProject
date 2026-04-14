"""Base recognizer interface.

Defines the abstract base class that all event recognizers must implement.
"""

from abc import ABC, abstractmethod

from f1_commentary.schemas.events import CandidateEvent, RaceStateSnapshot


class EventRecognizer(ABC):
    """Base class for all event recognizers."""

    @abstractmethod
    def recognize(self, snapshot: RaceStateSnapshot) -> list[CandidateEvent]:
        """Analyze a race state snapshot and return candidate events."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this recognizer."""
        ...
