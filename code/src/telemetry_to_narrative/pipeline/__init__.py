"""Phase 15 — end-to-end pipeline orchestration.

This module is deliberately boring: it wires together already-built
components (replay adapter, state engine, scheduler, generator, beat
manager, output adapters, baseline) into a single runnable pipeline.

No new logic lives here — stages are coordinated via ``PipelineRunner``,
with configurable stop points for debugging and a common trace writer
so every stage's intermediates land on disk as JSONL.
"""

from src.telemetry_to_narrative.pipeline.config import (
    PipelineConfig,
    PipelineMode,
    StopAfter,
)
from src.telemetry_to_narrative.pipeline.stage_result import (
    PipelineCounters,
    PipelineResult,
)
from src.telemetry_to_narrative.pipeline.trace_writer import TraceWriter
from src.telemetry_to_narrative.pipeline.pipeline_runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "PipelineMode",
    "StopAfter",
    "PipelineCounters",
    "PipelineResult",
    "TraceWriter",
    "PipelineRunner",
]
