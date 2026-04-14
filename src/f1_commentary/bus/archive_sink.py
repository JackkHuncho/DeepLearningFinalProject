"""Archive sink for completed session logs.

Writes trace outputs and artifacts to separate organized files,
one JSONL per event type within an archive directory.
"""

import logging
from pathlib import Path

from f1_commentary.schemas.events import BaseEvent

logger = logging.getLogger(__name__)


class ArchiveSinkConsumer:
    """Routes events to per-type archive files inside *archive_dir*.

    Each event type gets its own ``<event_type>.jsonl`` file.
    """

    def __init__(self, archive_dir: Path) -> None:
        self._archive_dir = Path(archive_dir)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    def handle_event(self, event: BaseEvent) -> None:
        """Append *event* to ``<archive_dir>/<event_type>.jsonl``."""
        dest = self._archive_dir / f"{event.event_type.value}.jsonl"
        line = event.model_dump_json()
        with open(dest, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.debug(
            "Archived %s to %s", event.event_type.value, dest
        )
