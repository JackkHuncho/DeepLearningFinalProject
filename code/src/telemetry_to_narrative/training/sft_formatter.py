"""SFT dataset formatter — converts aligned windows into training examples.

Constructs the ``input_text`` / ``target_text`` pairs that form the
supervised fine-tuning dataset.  The input is assembled from:

1. **Observed facts** — direct state facts from the snapshot (e.g.
   "VER is P1 on lap 18 with 0.8s lead").
2. **Derived inferences** — computed features that require interpretation
   (e.g. "HAM pace trend is improving", "overtake threat increasing").
3. **Retrieved context** — placeholder for Phase 6 retrieval augmentation.

The target is the aligned transcript text (cleaned commentary).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.telemetry_to_narrative.schemas.sft_dataset import (
    AlignmentConfidence,
    EventWindow,
    SFTExample,
    SourceRef,
    TranscriptSegment,
)
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.training.transcript_aligner import AlignmentResult

logger = logging.getLogger(__name__)


def _make_example_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Fact extraction ──────────────────────────────────────────────────────

def extract_observed_facts(window: EventWindow) -> list[str]:
    """Extract direct state facts from the window's state summary."""
    facts: list[str] = []
    summary = window.state_summary

    # Global context.
    lap = summary.get("lap")
    if lap is not None:
        facts.append(f"Current lap: {lap}")

    leader = summary.get("leader")
    if leader:
        facts.append(f"Race leader: {leader}")

    track_status = summary.get("track_status")
    if track_status:
        facts.append(f"Track status: {track_status}")

    if summary.get("safety_car"):
        facts.append("Safety car deployed")

    # Per-driver facts.
    drivers = summary.get("drivers", {})
    for abbr, ds in drivers.items():
        parts = []
        if ds.get("position") is not None:
            parts.append(f"P{ds['position']}")
        if ds.get("tire_compound"):
            compound = ds["tire_compound"]
            age = ds.get("tire_age_laps")
            tire_str = f"on {compound}" + (f" (lap {age})" if age is not None else "")
            parts.append(tire_str)
        if ds.get("gap_ahead_sec") is not None:
            parts.append(f"gap ahead {ds['gap_ahead_sec']:.1f}s")
        if ds.get("speed_kph") is not None:
            parts.append(f"speed {ds['speed_kph']:.0f} kph")

        if parts:
            facts.append(f"{abbr}: {', '.join(parts)}")

    # Battle facts.
    battles = summary.get("battles", [])
    for b in battles:
        gap = b.get("gap_sec")
        gap_str = f" (gap {gap:.2f}s)" if gap is not None else ""
        facts.append(f"Battle: {b['behind']} chasing {b['ahead']}{gap_str}")

    return facts


def extract_derived_inferences(window: EventWindow) -> list[str]:
    """Extract computed inferences from the window's state summary."""
    inferences: list[str] = []
    drivers = window.state_summary.get("drivers", {})

    for abbr, ds in drivers.items():
        # Pace trend.
        trend = ds.get("pace_trend")
        if trend is not None:
            if trend < -0.1:
                inferences.append(f"{abbr} pace is improving (trend {trend:.3f})")
            elif trend > 0.1:
                inferences.append(f"{abbr} pace is degrading (trend {trend:+.3f})")

        # Degradation.
        deg = ds.get("degradation_score")
        if deg is not None and deg > 0.5:
            inferences.append(f"{abbr} experiencing high tire degradation ({deg:.2f})")

        # Pit window.
        if ds.get("pit_window_open"):
            inferences.append(f"{abbr} pit window is open")

    # Battle intensity.
    battles = window.state_summary.get("battles", [])
    for b in battles:
        intensity = b.get("intensity")
        if intensity is not None and intensity > 0.6:
            inferences.append(
                f"High-intensity battle between {b['behind']} and {b['ahead']} "
                f"(intensity {intensity:.2f})"
            )

    # Event-type-specific inferences.
    etype = window.event_type.value
    if etype == "overtake":
        inferences.append("Position change detected — overtake likely")
    elif etype == "pit_strategy":
        inferences.append("Tire compound change — strategic pit stop")
    elif etype == "race_control":
        inferences.append("Race control intervention — track conditions changed")

    return inferences


# ── Input text formatting ────────────────────────────────────────────────

def format_input_text(
    facts: list[str],
    inferences: list[str],
    context: list[str],
    event_type: EventType,
    involved_drivers: list[str],
) -> str:
    """Format the structured components into a single prompt-ready input string.

    The format is designed to be clear and parseable by both humans and LLMs:

    ::

        [Event: overtake | Drivers: VER, HAM]

        ## Observed Facts
        - VER: P1, on MEDIUM (lap 12)
        - HAM: P2, gap ahead 0.8s
        ...

        ## Derived Inferences
        - HAM pace is improving (trend -0.15)
        ...

        ## Context
        (none available)
    """
    lines: list[str] = []

    # Header.
    driver_str = ", ".join(involved_drivers) if involved_drivers else "field"
    lines.append(f"[Event: {event_type.value} | Drivers: {driver_str}]")
    lines.append("")

    # Facts.
    lines.append("## Observed Facts")
    if facts:
        for f in facts:
            lines.append(f"- {f}")
    else:
        lines.append("- (no facts available)")
    lines.append("")

    # Inferences.
    lines.append("## Derived Inferences")
    if inferences:
        for inf in inferences:
            lines.append(f"- {inf}")
    else:
        lines.append("- (no inferences available)")
    lines.append("")

    # Context (placeholder for Phase 6).
    lines.append("## Context")
    if context:
        for c in context:
            lines.append(f"- {c}")
    else:
        lines.append("- (none available)")

    return "\n".join(lines)


def _clean_target_text(text: str) -> str:
    """Clean transcript text for use as training target.

    Removes leading/trailing whitespace and normalises internal whitespace.
    """
    return " ".join(text.split())


# ── Main formatter ───────────────────────────────────────────────────────

def format_sft_example(alignment: AlignmentResult) -> SFTExample:
    """Convert a single alignment result into an SFTExample.

    If the alignment is UNMATCHED (no transcript segment), the target_text
    will be empty.  These can be filtered out later or kept for audit.
    """
    window = alignment.window
    segment = alignment.segment

    facts = extract_observed_facts(window)
    inferences = extract_derived_inferences(window)
    context: list[str] = []  # Phase 6 placeholder

    input_text = format_input_text(
        facts, inferences, context,
        window.event_type, window.involved_drivers,
    )

    target_text = ""
    if segment is not None:
        target_text = _clean_target_text(segment.text)

    source_refs = [
        SourceRef(
            snapshot_frame_id=window.centre_frame_id,
            beat_id=window.window_id,
            transcript_segment_id=segment.segment_id if segment else None,
        )
    ]

    return SFTExample(
        example_id=_make_example_id(),
        session_info=window.session_info,
        event_type=window.event_type,
        storyline_id=window.storyline_id,
        involved_drivers=window.involved_drivers,
        observed_facts=facts,
        derived_inferences=inferences,
        retrieved_context=context,
        input_text=input_text,
        target_text=target_text,
        alignment_confidence=alignment.confidence,
        source_refs=source_refs,
    )


def format_sft_dataset(alignments: list[AlignmentResult]) -> list[SFTExample]:
    """Convert all alignment results into SFTExamples.

    Returns all examples including unmatched ones (for audit).
    """
    examples = [format_sft_example(a) for a in alignments]
    matched = sum(1 for e in examples if e.alignment_confidence != AlignmentConfidence.UNMATCHED)
    logger.info(
        "Formatted %d SFT examples (%d matched, %d unmatched).",
        len(examples), matched, len(examples) - matched,
    )
    return examples
