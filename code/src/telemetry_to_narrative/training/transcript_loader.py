"""Transcript ingestion module — loads commentary from multiple formats.

Supported formats
=================
- **JSON**: list of segment objects or ``{"segments": [...]}``
- **JSONL**: one segment object per line
- **CSV**: columns ``timestamp, text`` (plus optional ``speaker``, ``end_timestamp``,
  ``drivers_mentioned``, ``event_type_hint``)
- **Plain text**: lines matching ``[HH:MM:SS] text`` or ``HH:MM:SS - text``

All loaders normalise output to a list of ``TranscriptSegment`` objects with
UTC-aware timestamps.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

from src.telemetry_to_narrative.schemas.sft_dataset import TranscriptSegment

logger = logging.getLogger(__name__)

# ── Timestamp normalisation ──────────────────────────────────────────────

_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_HMS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")

# Plain-text line patterns:  [HH:MM:SS] text  or  HH:MM:SS - text
_TXT_BRACKET_RE = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\]\s*(.+)$")
_TXT_DASH_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2})\s*[-–]\s*(.+)$")


def parse_timestamp(raw: str, reference_date: Optional[datetime] = None) -> datetime:
    """Parse a timestamp string into a timezone-aware datetime.

    Supports ISO-8601, ``HH:MM:SS``, and epoch seconds.  When only a time
    component is given, *reference_date* supplies the date portion (defaults
    to 2024-01-01 UTC).
    """
    raw = raw.strip()

    # ISO-8601 (with or without timezone).
    if _ISO_RE.match(raw):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

    # HH:MM:SS only.
    m = _HMS_RE.match(raw)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ref = reference_date or datetime(2024, 1, 1, tzinfo=timezone.utc)
        return ref.replace(hour=h, minute=mn, second=s, microsecond=0)

    # Epoch seconds (int or float).
    try:
        epoch = float(raw)
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except ValueError:
        pass

    raise ValueError(f"Cannot parse timestamp: {raw!r}")


def _make_segment_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Format-specific loaders ─────────────────────────────────────────────

def _segment_from_dict(d: dict, reference_date: Optional[datetime] = None) -> TranscriptSegment:
    """Build a TranscriptSegment from a raw dict (JSON/JSONL row)."""
    ts = parse_timestamp(str(d["timestamp"]), reference_date)

    end_ts = None
    if d.get("end_timestamp"):
        end_ts = parse_timestamp(str(d["end_timestamp"]), reference_date)

    drivers = d.get("drivers_mentioned", [])
    if isinstance(drivers, str):
        drivers = [x.strip() for x in drivers.split(",") if x.strip()]

    return TranscriptSegment(
        segment_id=d.get("segment_id") or _make_segment_id(),
        timestamp=ts,
        end_timestamp=end_ts,
        speaker=d.get("speaker"),
        text=str(d["text"]),
        drivers_mentioned=drivers,
        event_type_hint=d.get("event_type_hint"),
    )


def load_json(path: Path, reference_date: Optional[datetime] = None) -> list[TranscriptSegment]:
    """Load transcript segments from a JSON file.

    Accepts either a bare list ``[{...}, ...]`` or an object with a
    ``"segments"`` key.
    """
    with open(path) as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        data = data.get("segments", [])
    if not isinstance(data, list):
        raise ValueError(f"Expected list or {{segments: [...]}} in {path}")

    segments = [_segment_from_dict(d, reference_date) for d in data]
    logger.info("Loaded %d segments from JSON: %s", len(segments), path)
    return segments


def load_jsonl(path: Path, reference_date: Optional[datetime] = None) -> list[TranscriptSegment]:
    """Load transcript segments from a JSONL file (one JSON object per line)."""
    segments: list[TranscriptSegment] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            segments.append(_segment_from_dict(d, reference_date))
    logger.info("Loaded %d segments from JSONL: %s", len(segments), path)
    return segments


def load_csv(path: Path, reference_date: Optional[datetime] = None) -> list[TranscriptSegment]:
    """Load transcript segments from a CSV file.

    Required columns: ``timestamp``, ``text``.
    Optional columns: ``segment_id``, ``end_timestamp``, ``speaker``,
    ``drivers_mentioned`` (comma-separated within field), ``event_type_hint``.
    """
    segments: list[TranscriptSegment] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            segments.append(_segment_from_dict(row, reference_date))
    logger.info("Loaded %d segments from CSV: %s", len(segments), path)
    return segments


def load_plain_text(
    path: Path,
    reference_date: Optional[datetime] = None,
) -> list[TranscriptSegment]:
    """Load transcript from a plain-text file with timestamped lines.

    Recognised formats per line::

        [HH:MM:SS] commentary text here
        HH:MM:SS - commentary text here

    Lines that don't match either pattern are skipped with a warning.
    """
    segments: list[TranscriptSegment] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            m = _TXT_BRACKET_RE.match(line) or _TXT_DASH_RE.match(line)
            if not m:
                logger.warning("Skipping unrecognised line %d in %s: %s", lineno, path, line[:80])
                continue

            time_str, text = m.group(1), m.group(2)
            ts = parse_timestamp(time_str, reference_date)
            segments.append(
                TranscriptSegment(
                    segment_id=_make_segment_id(),
                    timestamp=ts,
                    text=text,
                )
            )

    logger.info("Loaded %d segments from plain text: %s", len(segments), path)
    return segments


# ── Unified loader ───────────────────────────────────────────────────────

_EXT_LOADERS = {
    ".json": load_json,
    ".jsonl": load_jsonl,
    ".csv": load_csv,
    ".txt": load_plain_text,
}


def load_transcript(
    path: str | Path,
    reference_date: Optional[datetime] = None,
) -> list[TranscriptSegment]:
    """Auto-detect format by extension and load transcript segments.

    Parameters
    ----------
    path : str | Path
        File path to the transcript.
    reference_date : datetime | None
        Date used to anchor ``HH:MM:SS``-only timestamps.

    Returns
    -------
    list[TranscriptSegment]
        Segments sorted by timestamp.
    """
    p = Path(path)
    loader = _EXT_LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported transcript format '{p.suffix}'. "
            f"Supported: {', '.join(_EXT_LOADERS)}"
        )
    segments = loader(p, reference_date)
    segments.sort(key=lambda s: s.timestamp)
    return segments
