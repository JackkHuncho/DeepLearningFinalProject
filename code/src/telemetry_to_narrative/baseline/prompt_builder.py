"""Baseline prompt builder — fair, structured prompt without system advantages.

Fairness design
===============
The baseline receives:
  - same observed facts (from ``extract_observed_facts``)
  - same derived inferences (from ``extract_derived_inferences``)
  - same event type + involved drivers header

The baseline does NOT receive:
  - storyline context (progression-aware continuity is a local-system feature)
  - retrieved context (gated retrieval is a local-system feature)
  - short-budget instruction variants (backpressure is a local-system feature)
  - beat manager or grounding guard post-processing

This gives both systems the same structured race-state input, then asks
each to produce one short commentary line.  Any quality difference in the
A/B comparison is attributable to system design, not to information asymmetry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.sft_dataset import EventWindow
from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo
from src.telemetry_to_narrative.training.event_window_builder import _build_state_summary
from src.telemetry_to_narrative.training.sft_formatter import (
    extract_observed_facts,
    extract_derived_inferences,
    format_input_text,
)

logger = logging.getLogger(__name__)


# Instruction for the baseline — deliberately comparable to the local
# system's default instruction so both models get the same ask.
_BASELINE_INSTRUCTION = (
    "Generate one short, professional-style commentary line grounded "
    "in the provided race state. Keep it to one or two sentences."
)


def build_baseline_prompt(
    snapshot: RaceStateSnapshot,
    beat: ScheduledBeat,
) -> str:
    """Build a baseline prompt from the same structured input as the local system.

    Mirrors the local runtime's fact/inference extraction so the baseline
    sees identical observed facts and derived inferences, but omits the
    storyline section and retrieval section used by the local system.

    Parameters
    ----------
    snapshot : RaceStateSnapshot
        Current race state.
    beat : ScheduledBeat
        The beat / event context.

    Returns
    -------
    prompt_text : str
        The full baseline prompt.
    """
    state_summary = _build_state_summary(snapshot, beat.involved_drivers)

    window = EventWindow(
        window_id="baseline",
        frame_ids=[snapshot.frame_id],
        timestamps=[snapshot.timestamp],
        centre_frame_id=snapshot.frame_id,
        centre_timestamp=snapshot.timestamp,
        session_info=snapshot.session_info,
        event_type=beat.event_type,
        storyline_id=beat.storyline_id,
        involved_drivers=beat.involved_drivers,
        priority_score=beat.priority_score,
        beat_description=beat.description,
        state_summary=state_summary,
    )

    facts = extract_observed_facts(window)
    inferences = extract_derived_inferences(window)

    # Empty context list — baseline has no retrieval.
    input_text = format_input_text(
        facts, inferences, [],
        beat.event_type, beat.involved_drivers,
    )

    # Deliberately NO storyline section — that's a local-system advantage.
    task_section = f"\n## Task\n{_BASELINE_INSTRUCTION}\n"

    return input_text + task_section


def build_baseline_prompt_from_state_summary(
    state_summary: dict,
    event_type,  # EventType
    involved_drivers: list[str],
) -> str:
    """Build a baseline prompt directly from a pre-built state summary.

    Useful when the summary is already computed and a full snapshot is
    unavailable (e.g. in offline evaluation from EventWindow records).
    """
    stub_ts = datetime.now(timezone.utc)
    stub_session = SessionInfo(year=0, grand_prix="unknown", session_type="R")
    window = EventWindow(
        window_id="baseline",
        frame_ids=[0],
        timestamps=[stub_ts],
        centre_frame_id=0,
        centre_timestamp=stub_ts,
        session_info=stub_session,
        event_type=event_type,
        involved_drivers=involved_drivers,
        state_summary=state_summary,
    )
    facts = extract_observed_facts(window)
    inferences = extract_derived_inferences(window)
    input_text = format_input_text(
        facts, inferences, [], event_type, involved_drivers,
    )
    task_section = f"\n## Task\n{_BASELINE_INSTRUCTION}\n"
    return input_text + task_section
