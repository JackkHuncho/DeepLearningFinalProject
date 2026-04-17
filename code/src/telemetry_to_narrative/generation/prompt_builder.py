"""Prompt builder — constructs inference prompts aligned with SFT training format.

The prompt reuses the exact same ``format_input_text`` structure from
the SFT formatter so that training and inference inputs are identical.
An additional generation instruction is appended at the end.

Prompt structure
================
::

    [Event: lead_battle | Drivers: VER, HAM]

    ## Observed Facts
    - VER: P1, on MEDIUM (lap 12), gap ahead 0.0s
    - HAM: P2, gap ahead 0.8s
    ...

    ## Derived Inferences
    - HAM pace is improving (trend -0.150)
    ...

    ## Context
    - (none available)

    ## Storyline
    - battle_HAM_VER (ongoing)

    ## Task
    Generate one short, professional-style commentary line grounded
    in the provided race state. Keep it to one or two sentences.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.training.event_window_builder import _build_state_summary
from src.telemetry_to_narrative.training.sft_formatter import (
    extract_observed_facts,
    extract_derived_inferences,
    format_input_text,
)
from src.telemetry_to_narrative.schemas.sft_dataset import EventWindow

logger = logging.getLogger(__name__)

# Generation instruction appended to every prompt.
_GENERATION_INSTRUCTION = (
    "Generate one short, professional-style commentary line grounded "
    "in the provided race state. Keep it to one or two sentences."
)

_SHORT_GENERATION_INSTRUCTION = (
    "Generate one concise commentary sentence grounded in the race state."
)


def build_prompt(
    snapshot: RaceStateSnapshot,
    beat: ScheduledBeat,
    retrieved_context: Optional[list[str]] = None,
    short_budget: Optional[bool] = None,
) -> tuple[str, bool]:
    """Build an inference prompt from a snapshot and scheduled beat.

    Parameters
    ----------
    snapshot : RaceStateSnapshot
        Current race state.
    beat : ScheduledBeat
        The beat selected for generation.
    retrieved_context : list[str] | None
        Optional retrieval-augmented context strings.
    short_budget : bool | None
        If True, use a shorter generation instruction (for backpressure).
        If None (default), inferred from ``beat.output_length_budget < 100``.

    Returns
    -------
    (prompt_text, retrieval_used)
        The formatted prompt and whether retrieval context was included.
    """
    # Infer short_budget from beat if not explicitly provided.
    if short_budget is None:
        short_budget = beat.output_length_budget < 100

    # Build a state summary dict in the same format as training.
    state_summary = _build_state_summary(snapshot, beat.involved_drivers)

    # Create a temporary EventWindow to reuse the SFT extraction functions.
    window = EventWindow(
        window_id="runtime",
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

    # Extract structured components using the same functions as training.
    facts = extract_observed_facts(window)
    inferences = extract_derived_inferences(window)

    # Retrieval context — fall back gracefully if unavailable.
    context: list[str] = []
    retrieval_used = False
    if retrieved_context:
        context = retrieved_context
        retrieval_used = True

    # Build the core input text (matches SFT format exactly).
    input_text = format_input_text(
        facts, inferences, context,
        beat.event_type, beat.involved_drivers,
    )

    # Append storyline section.
    storyline_section = "\n## Storyline\n"
    if beat.storyline_id:
        storyline_section += f"- {beat.storyline_id} (ongoing)\n"
    else:
        storyline_section += "- (no active storyline)\n"

    # Append generation instruction.
    instruction = _SHORT_GENERATION_INSTRUCTION if short_budget else _GENERATION_INSTRUCTION
    task_section = f"\n## Task\n{instruction}\n"

    prompt = input_text + storyline_section + task_section

    return prompt, retrieval_used


def build_prompt_from_state_summary(
    state_summary: dict,
    event_type: EventType,
    involved_drivers: list[str],
    storyline_id: Optional[str] = None,
    retrieved_context: Optional[list[str]] = None,
    short_budget: bool = False,
) -> tuple[str, bool]:
    """Build a prompt directly from a pre-built state summary dict.

    Useful when you already have the summary (e.g. from a loaded EventWindow)
    and don't need a full snapshot.
    """
    window = EventWindow(
        window_id="runtime",
        frame_ids=[0],
        timestamps=[],
        centre_frame_id=0,
        centre_timestamp=None,  # type: ignore
        session_info=None,  # type: ignore
        event_type=event_type,
        involved_drivers=involved_drivers,
        state_summary=state_summary,
    )

    facts = extract_observed_facts(window)
    inferences = extract_derived_inferences(window)

    context: list[str] = []
    retrieval_used = False
    if retrieved_context:
        context = retrieved_context
        retrieval_used = True

    input_text = format_input_text(
        facts, inferences, context, event_type, involved_drivers,
    )

    storyline_section = "\n## Storyline\n"
    if storyline_id:
        storyline_section += f"- {storyline_id} (ongoing)\n"
    else:
        storyline_section += "- (no active storyline)\n"

    instruction = _SHORT_GENERATION_INSTRUCTION if short_budget else _GENERATION_INSTRUCTION
    task_section = f"\n## Task\n{instruction}\n"

    return input_text + storyline_section + task_section, retrieval_used
