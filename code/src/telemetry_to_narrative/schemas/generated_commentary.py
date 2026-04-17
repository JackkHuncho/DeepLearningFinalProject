"""GeneratedCommentary schema — output of the runtime commentary generator.

Each object captures the full provenance chain from prompt through
model output, enabling qualitative analysis, latency tracking, and
paper examples.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.candidate_event import EventType


class GenerationConfig(BaseModel):
    """Snapshot of generation hyper-parameters used for one inference call."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    backend: str = "mock"


class GeneratedCommentary(BaseModel):
    """One generated commentary output with full provenance."""

    generation_id: str
    timestamp: datetime
    beat_id: str
    storyline_id: Optional[str] = None
    event_type: EventType
    involved_drivers: list[str] = Field(default_factory=list)

    # Model info.
    model_variant: str = Field(
        description="Model identifier, e.g. 'mistral-7b', 'mistral-7b+sft-adapter', 'mock'",
    )

    # Prompt and output.
    prompt_text: str = Field(
        description="Full prompt sent to the model — always logged for debugging",
    )
    raw_output_text: str = Field(
        description="Raw text returned by the model (before any post-processing)",
    )
    commentary_text: str = Field(
        default="",
        description="Post-processed commentary (trimmed, cleaned)",
    )

    # Metrics.
    output_length: int = Field(
        default=0,
        description="Length of raw_output_text in characters",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock inference time in milliseconds",
    )

    # Context flags.
    retrieved_context_used: bool = Field(
        default=False,
        description="Whether retrieval context was included in the prompt",
    )
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    model_config = {"ser_json_timedelta": "float"}
