"""PipelineResult + counters — inspectable record of one run.

Every intermediate list is preserved on the returned object so the
caller can inspect post-hoc without re-running the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.candidate_event import CandidateEvent
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.final_commentary import FinalCommentary
from src.telemetry_to_narrative.schemas.baseline_commentary import BaselineCommentary


@dataclass
class PipelineCounters:
    """Summary counters for end-of-run reporting."""

    frames: int = 0
    snapshots: int = 0
    candidate_events: int = 0
    beats_total: int = 0
    beats_selected: int = 0
    beats_suppressed: int = 0
    generations: int = 0
    finals_emitted: int = 0
    finals_suppressed: int = 0
    baselines: int = 0
    fallbacks_used: int = 0
    total_gen_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "snapshots": self.snapshots,
            "candidate_events": self.candidate_events,
            "beats_total": self.beats_total,
            "beats_selected": self.beats_selected,
            "beats_suppressed": self.beats_suppressed,
            "generations": self.generations,
            "finals_emitted": self.finals_emitted,
            "finals_suppressed": self.finals_suppressed,
            "baselines": self.baselines,
            "fallbacks_used": self.fallbacks_used,
            "avg_gen_latency_ms": (
                round(self.total_gen_latency_ms / self.generations, 2)
                if self.generations else 0.0
            ),
            "total_gen_latency_ms": round(self.total_gen_latency_ms, 2),
        }


@dataclass
class PipelineResult:
    """Everything the runner produced during one run, in memory."""

    frames: list[TelemetryFrame] = field(default_factory=list)
    snapshots: list[RaceStateSnapshot] = field(default_factory=list)
    candidates: list[CandidateEvent] = field(default_factory=list)
    beats: list[ScheduledBeat] = field(default_factory=list)
    generated: list[GeneratedCommentary] = field(default_factory=list)
    finals: list[FinalCommentary] = field(default_factory=list)
    baselines: list[BaselineCommentary] = field(default_factory=list)
    counters: PipelineCounters = field(default_factory=PipelineCounters)
    trace_paths: dict[str, str] = field(default_factory=dict)
    stopped_at: Optional[str] = None
