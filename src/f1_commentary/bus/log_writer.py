"""Append-only JSONL log writer for event persistence.

Subscribes to the event bus and writes every event to a JSONL file
for deterministic replay and debugging.
"""

import json
import logging
from pathlib import Path

from f1_commentary.schemas.events import BaseEvent, EventType

logger = logging.getLogger(__name__)


class JSONLLogWriter:
    """Append-only JSONL writer for pipeline events.

    Each call to :meth:`write` serializes a :class:`BaseEvent` to JSON and
    appends it as a single line in the log file.
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._log_path

    def write(self, event: BaseEvent) -> None:
        """Append *event* as a JSON line."""
        line = event.model_dump_json()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.debug("Wrote %s to %s", event.event_type.value, self._log_path)

    def read_all(self) -> list[dict]:
        """Read all events from the log and return them as dicts."""
        if not self._log_path.exists():
            return []
        events: list[dict] = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def read_by_type(self, event_type: EventType) -> list[dict]:
        """Read events filtered to a specific *event_type*."""
        return [
            e for e in self.read_all() if e.get("event_type") == event_type.value
        ]
