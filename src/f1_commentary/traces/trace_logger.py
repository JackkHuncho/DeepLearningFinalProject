"""Structured per-beat trace logger for auditability and reproducibility."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class TraceEntry(BaseModel):
    """One complete trace record for a single commentary beat."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    beat_id: str = ""
    event_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Pipeline stage snapshots
    telemetry_snapshot: dict = Field(default_factory=dict)
    derived_features: dict = Field(default_factory=dict)
    candidate_event: dict = Field(default_factory=dict)
    scheduled_beat: dict = Field(default_factory=dict)
    retrieval_context: dict = Field(default_factory=dict)

    # Output stages
    storyline_id: Optional[str] = None
    generated_commentary: dict = Field(default_factory=dict)
    guard_decision: dict = Field(default_factory=dict)
    final_commentary: dict = Field(default_factory=dict)

    # Metadata
    model_variant: str = ""
    latency_breakdown: dict = Field(default_factory=dict)  # stage -> ms
    total_latency_ms: float = 0.0


class TraceLogger:
    """Append-only JSONL trace logger."""

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._session_file: Optional[Path] = None

    def start_session(self, session_key: str) -> Path:
        """Start a new trace session, returns path to trace file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_file = self.trace_dir / f"trace_{session_key}_{timestamp}.jsonl"
        return self._session_file

    def log(self, entry: TraceEntry) -> None:
        """Append a trace entry to the current session file."""
        if self._session_file is None:
            raise RuntimeError("No active session. Call start_session() first.")
        with open(self._session_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def read_traces(self, trace_file: Path) -> list[TraceEntry]:
        """Read all trace entries from a file."""
        entries = []
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(TraceEntry.model_validate_json(line))
        return entries
