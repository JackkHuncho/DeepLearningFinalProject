"""State rebuild from persisted event logs.

Replays a JSONL log file to reconstruct pipeline state at any point in time.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from f1_commentary.schemas.events import EventType

logger = logging.getLogger(__name__)


class StateRebuildConsumer:
    """Rebuilds pipeline state from a JSONL operational log.

    Reads the log line-by-line and accumulates counts, latest snapshots,
    and key summaries into a state dictionary.
    """

    def rebuild(self, log_path: Path) -> dict[str, Any]:
        """Read JSONL at *log_path*, replay events, return reconstructed state dict.

        The returned dict contains:
        - ``total_events``: total number of events replayed
        - ``event_counts``: mapping of event_type -> count
        - ``last_telemetry_lap``: lap number from the most recent telemetry frame
        - ``last_session_key``: session key from the most recent telemetry/race-state event
        - ``final_commentaries``: list of final commentary texts in order
        - ``success``: True if rebuild completed without error
        """
        if not Path(log_path).exists():
            logger.warning("Log file does not exist: %s", log_path)
            return {"total_events": 0, "event_counts": {}, "success": False}

        event_counts: Counter[str] = Counter()
        last_telemetry_lap: int | None = None
        last_session_key: str | None = None
        final_commentaries: list[str] = []

        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                etype = record.get("event_type", "unknown")
                event_counts[etype] += 1

                if etype == EventType.TELEMETRY_FRAME.value:
                    last_telemetry_lap = record.get("lap", last_telemetry_lap)
                    last_session_key = record.get(
                        "session_key", last_session_key
                    )
                elif etype == EventType.RACE_STATE_UPDATE.value:
                    last_session_key = record.get(
                        "session_key", last_session_key
                    )
                elif etype == EventType.FINAL_COMMENTARY.value:
                    text = record.get("final_text", "")
                    if text:
                        final_commentaries.append(text)

        return {
            "total_events": sum(event_counts.values()),
            "event_counts": dict(event_counts),
            "last_telemetry_lap": last_telemetry_lap,
            "last_session_key": last_session_key,
            "final_commentaries": final_commentaries,
            "success": True,
        }
