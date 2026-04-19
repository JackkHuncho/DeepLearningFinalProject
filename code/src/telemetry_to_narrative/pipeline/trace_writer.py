"""TraceWriter — buffered JSONL writer for every pipeline stage.

One instance per run.  Opens a fixed set of output files under a given
trace directory, writes one JSON record per line, and closes them on
context exit.  Missing trace directory = no-op writers (useful for
tests and ad-hoc runs).

Schemas are Pydantic models, so each record uses ``model_dump_json()``
to avoid ad-hoc serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Stage name → filename on disk.  These names flow into
# ``PipelineResult.trace_paths`` so callers can find them.
_FILE_MAP: dict[str, str] = {
    "frames":      "replay_frames.jsonl",
    "snapshots":   "snapshots.jsonl",
    "candidates":  "candidate_events.jsonl",
    "beats":       "scheduled_beats.jsonl",
    "generated":   "generated_commentary.jsonl",
    "finals":      "final_commentary.jsonl",
    "baselines":   "baseline_commentary.jsonl",
    "summary":     "run_summary.json",
}


class TraceWriter:
    """Manages JSONL output for one pipeline run.

    Usage::

        with TraceWriter("data/artifacts/run_42") as tw:
            tw.write("frames", telemetry_frame)
            ...
            tw.write_summary(result_dict)

    If ``base_dir`` is None the writer becomes a no-op.
    """

    def __init__(self, base_dir: Optional[str | Path]) -> None:
        self.base_dir: Optional[Path] = Path(base_dir) if base_dir else None
        self._files: dict[str, IO] = {}
        self._counts: dict[str, int] = {k: 0 for k in _FILE_MAP}
        self._paths: dict[str, str] = {}

    # ── Context manager ─────────────────────────────────────────────

    def __enter__(self) -> "TraceWriter":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── File lifecycle ──────────────────────────────────────────────

    def _open(self) -> None:
        if self.base_dir is None:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for stage, fname in _FILE_MAP.items():
            if stage == "summary":
                # Summary is a single JSON, not JSONL; written at close.
                self._paths[stage] = str(self.base_dir / fname)
                continue
            path = self.base_dir / fname
            self._files[stage] = open(path, "w")
            self._paths[stage] = str(path)
        logger.info("TraceWriter opened %d files under %s", len(self._files), self.base_dir)

    def close(self) -> None:
        for stage, fh in self._files.items():
            try:
                fh.close()
            except Exception as exc:
                logger.warning("Failed to close trace file for %s: %s", stage, exc)
        self._files.clear()

    # ── Writing ─────────────────────────────────────────────────────

    def write(self, stage: str, record: BaseModel) -> None:
        """Append one Pydantic record to the stage's JSONL file."""
        if stage not in _FILE_MAP:
            raise KeyError(f"Unknown trace stage '{stage}'. Known: {list(_FILE_MAP)}")
        fh = self._files.get(stage)
        if fh is None:
            return  # no-op mode
        fh.write(record.model_dump_json() + "\n")
        self._counts[stage] += 1

    def write_many(self, stage: str, records) -> None:
        for r in records:
            self.write(stage, r)

    def write_summary(self, summary: dict) -> None:
        """Write the final run summary as a single JSON blob."""
        if self.base_dir is None:
            return
        import json
        path = self.base_dir / _FILE_MAP["summary"]
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Wrote run summary to %s", path)

    # ── Introspection ───────────────────────────────────────────────

    @property
    def paths(self) -> dict[str, str]:
        """Stage → path mapping (empty in no-op mode)."""
        return dict(self._paths)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
