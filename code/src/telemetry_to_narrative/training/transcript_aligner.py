"""Transcript alignment — match transcript segments to event windows.

Two alignment modes
===================
- **Strict**: timestamp must fall within the event window's time range.
- **Loose**: allows a configurable tolerance beyond the window edges, and
  falls back to heuristic matching by driver mentions / event-type hints.

The aligner assigns an ``AlignmentConfidence`` level to each match.
Unmatched windows are preserved (with ``UNMATCHED`` confidence) for audit.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from src.telemetry_to_narrative.schemas.sft_dataset import (
    AlignmentConfidence,
    EventWindow,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# Default tolerance for "near" matching (seconds).
DEFAULT_TOLERANCE_SEC = 10.0


class AlignmentResult:
    """Container for a single window's alignment outcome."""

    __slots__ = ("window", "segment", "confidence")

    def __init__(
        self,
        window: EventWindow,
        segment: Optional[TranscriptSegment],
        confidence: AlignmentConfidence,
    ) -> None:
        self.window = window
        self.segment = segment
        self.confidence = confidence

    def __repr__(self) -> str:
        seg_id = self.segment.segment_id if self.segment else "None"
        return (
            f"AlignmentResult(window={self.window.window_id}, "
            f"segment={seg_id}, confidence={self.confidence.value})"
        )


def _timestamp_in_window(segment: TranscriptSegment, window: EventWindow) -> bool:
    """Check if the segment timestamp falls inside the window's time range."""
    if not window.timestamps:
        return False
    win_start = min(window.timestamps)
    win_end = max(window.timestamps)
    return win_start <= segment.timestamp <= win_end


def _timestamp_near_window(
    segment: TranscriptSegment,
    window: EventWindow,
    tolerance: timedelta,
) -> bool:
    """Check if the segment timestamp is within tolerance of the window."""
    if not window.timestamps:
        return False
    win_start = min(window.timestamps)
    win_end = max(window.timestamps)
    return (win_start - tolerance) <= segment.timestamp <= (win_end + tolerance)


def _heuristic_match_score(segment: TranscriptSegment, window: EventWindow) -> float:
    """Score a heuristic match based on driver mentions and event-type hints.

    Returns a score in [0, 1].  0 means no heuristic evidence of a match.
    """
    score = 0.0

    # Driver overlap.
    if segment.drivers_mentioned and window.involved_drivers:
        seg_drivers = set(segment.drivers_mentioned)
        win_drivers = set(window.involved_drivers)
        overlap = len(seg_drivers & win_drivers)
        if overlap > 0:
            score += 0.5 * (overlap / max(len(win_drivers), 1))

    # Event-type hint match.
    if segment.event_type_hint and window.event_type:
        hint = segment.event_type_hint.lower().replace(" ", "_")
        etype = window.event_type.value.lower()
        if hint == etype or hint in etype or etype in hint:
            score += 0.5

    return min(1.0, score)


def align_transcripts(
    windows: list[EventWindow],
    segments: list[TranscriptSegment],
    mode: str = "loose",
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    heuristic_threshold: float = 0.3,
) -> list[AlignmentResult]:
    """Align transcript segments to event windows.

    Parameters
    ----------
    windows : list[EventWindow]
        Event windows to match against.
    segments : list[TranscriptSegment]
        Transcript segments (should be sorted by timestamp).
    mode : str
        ``"strict"`` or ``"loose"``.
    tolerance_sec : float
        Seconds of tolerance for "near" matching (loose mode only).
    heuristic_threshold : float
        Minimum heuristic score to accept a match (loose mode only).

    Returns
    -------
    list[AlignmentResult]
        One result per window, preserving window order.
    """
    if mode not in ("strict", "loose"):
        raise ValueError(f"mode must be 'strict' or 'loose', got {mode!r}")

    tolerance = timedelta(seconds=tolerance_sec)
    results: list[AlignmentResult] = []
    used_segments: set[str] = set()  # prevent double-assignment

    for window in windows:
        best_segment: Optional[TranscriptSegment] = None
        best_confidence = AlignmentConfidence.UNMATCHED
        best_heuristic_score = 0.0

        for segment in segments:
            if segment.segment_id in used_segments:
                continue

            # 1. Exact: timestamp inside window.
            if _timestamp_in_window(segment, window):
                # Among exact matches, prefer the one with best heuristic too.
                h_score = _heuristic_match_score(segment, window)
                if best_confidence != AlignmentConfidence.EXACT or h_score > best_heuristic_score:
                    best_segment = segment
                    best_confidence = AlignmentConfidence.EXACT
                    best_heuristic_score = h_score
                continue

            if mode == "strict":
                continue

            # 2. Near: within tolerance.
            if _timestamp_near_window(segment, window, tolerance):
                if best_confidence not in (AlignmentConfidence.EXACT,):
                    h_score = _heuristic_match_score(segment, window)
                    if best_confidence != AlignmentConfidence.NEAR or h_score > best_heuristic_score:
                        best_segment = segment
                        best_confidence = AlignmentConfidence.NEAR
                        best_heuristic_score = h_score
                continue

            # 3. Heuristic: driver/event-type match only.
            if best_confidence not in (AlignmentConfidence.EXACT, AlignmentConfidence.NEAR):
                h_score = _heuristic_match_score(segment, window)
                if h_score >= heuristic_threshold and h_score > best_heuristic_score:
                    best_segment = segment
                    best_confidence = AlignmentConfidence.HEURISTIC
                    best_heuristic_score = h_score

        if best_segment is not None:
            used_segments.add(best_segment.segment_id)

        results.append(AlignmentResult(window, best_segment, best_confidence))

    # Log summary.
    counts = {c: 0 for c in AlignmentConfidence}
    for r in results:
        counts[r.confidence] += 1
    logger.info(
        "Alignment complete: %d windows → exact=%d, near=%d, heuristic=%d, unmatched=%d",
        len(results), counts[AlignmentConfidence.EXACT],
        counts[AlignmentConfidence.NEAR], counts[AlignmentConfidence.HEURISTIC],
        counts[AlignmentConfidence.UNMATCHED],
    )
    return results
