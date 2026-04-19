"""BaselineCommentary schema — output of the frontier LLM baseline.

Mirrors GeneratedCommentary closely so baseline outputs can flow through
the same trace/evaluation pipeline, but without pipeline-specific fields
(retrieved_context_used, etc.) that don't apply to the baseline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.telemetry_to_narrative.schemas.candidate_event import EventType


class BaselineGenerationConfig(BaseModel):
    """Hyper-parameters for one baseline LLM call."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    provider: str = "mock"
    model_name: str = "mock"


class BaselineCommentary(BaseModel):
    """One baseline commentary output with full provenance.

    This schema is intentionally similar to GeneratedCommentary so that
    outputs from both systems can be logged, compared, and evaluated
    with the same tooling.
    """

    baseline_id: str
    timestamp: datetime
    beat_id: str
    storyline_id: Optional[str] = None
    event_type: EventType
    involved_drivers: list[str] = Field(default_factory=list)

    # Provider / model info.
    model_name: str = Field(
        description="Model identifier, e.g. 'gpt-4o-mini', 'claude-3-haiku', 'mock'",
    )
    provider: str = Field(
        default="mock",
        description="Provider family: 'openai', 'anthropic', 'google', 'mock'",
    )

    # Prompt and output.
    prompt_text: str = Field(
        description="Full prompt sent to the model — always logged for paper/debug",
    )
    raw_output_text: str = Field(
        description="Raw text returned by the model before minimal cleanup",
    )
    commentary_text: str = Field(
        default="",
        description="Post-processed commentary (minimal cleanup only: trim + unquote)",
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

    generation_config: BaselineGenerationConfig = Field(
        default_factory=BaselineGenerationConfig,
    )

    # Comparison-friendly marker so downstream eval pipelines can route.
    system: str = Field(
        default="baseline",
        description="Always 'baseline' — distinguishes from 'local' runtime output",
    )

    model_config = {"ser_json_timedelta": "float"}
