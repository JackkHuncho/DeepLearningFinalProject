"""PipelineRunner — orchestration-only end-to-end runner.

Wires together (in order):

    frame source  →  state engine  →  candidate extractor
                  →  editorial scheduler
                  →  commentary generator  (local mode)
                  |  baseline generator    (baseline mode)
                  →  beat manager          (local mode only)
                  →  output adapters

All stages share the same stage-keyed ``TraceWriter``.  IDs are
preserved end-to-end:

    frame_id  →  snapshot.frame_id
              →  candidate.source_frame_id
              →  beat.source_frame_id / beat.beat_id
              →  generation.beat_id + generation_id
              →  final.beat_id + source_generation_id + final_id
              (baseline.beat_id + baseline_id in baseline mode)

The runner accepts either:
  - a pre-built iterable of ``TelemetryFrame`` objects, OR
  - a FastF1 session triplet (year, gp, session_type) that it will
    load lazily through ``ReplayAdapter``.

This keeps the runner testable without FastF1.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary,
    GenerationConfig,
)
from src.telemetry_to_narrative.schemas.baseline_commentary import (
    BaselineCommentary,
    BaselineGenerationConfig,
)

from src.telemetry_to_narrative.state.race_state_engine import RaceStateEngine
from src.telemetry_to_narrative.state.feature_extractor import FeatureExtractor

from src.telemetry_to_narrative.scheduler.editorial_scheduler import EditorialScheduler
from src.telemetry_to_narrative.scheduler.snapshot_to_candidates import extract_candidates

from src.telemetry_to_narrative.generation.commentary_generator import CommentaryGenerator
from src.telemetry_to_narrative.generation.model_loader import load_backend

from src.telemetry_to_narrative.beat_manager.beat_manager import BeatManager
from src.telemetry_to_narrative.beat_manager.output_adapters import (
    SSEAdapter,
    TerminalAdapter,
)

from src.telemetry_to_narrative.baseline.backends import load_baseline_backend
from src.telemetry_to_narrative.baseline.baseline_generator import BaselineGenerator

from src.telemetry_to_narrative.pipeline.config import (
    PipelineConfig,
    PipelineMode,
    Scenario,
    StopAfter,
)
from src.telemetry_to_narrative.pipeline.stage_result import (
    PipelineCounters,
    PipelineResult,
)
from src.telemetry_to_narrative.pipeline.trace_writer import TraceWriter

logger = logging.getLogger(__name__)


# ── Scenario filter ──────────────────────────────────────────────────────

_SCENARIO_EVENTS: dict[Scenario, set[EventType]] = {
    Scenario.LEAD_BATTLE: {EventType.LEAD_BATTLE, EventType.OVERTAKE},
    Scenario.PIT_STRATEGY: {EventType.PIT_STRATEGY},
    Scenario.RACE_CONTROL: {EventType.RACE_CONTROL},
}


def _apply_scenario_filter(candidates, scenario: Scenario):
    """Return only candidates matching the chosen scenario."""
    if scenario == Scenario.ALL:
        return candidates
    allowed = _SCENARIO_EVENTS.get(scenario, set())
    return [c for c in candidates if c.event_type in allowed]


# ── Runner ───────────────────────────────────────────────────────────────

class PipelineRunner:
    """Connects existing modules into a single end-to-end run.

    Does not redesign any component.  Controls are limited to which
    stages execute and where intermediates are written.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ── Public API ─────────────────────────────────────────────────

    def run(
        self,
        frames: Optional[Iterable[TelemetryFrame]] = None,
        *,
        fastf1_session: Optional[tuple[int, str, str]] = None,
        cache_dir: Optional[str] = None,
    ) -> PipelineResult:
        """Run the pipeline.

        Exactly one of ``frames`` or ``fastf1_session`` must be provided.
        """
        self.config.validate()
        if (frames is None) == (fastf1_session is None):
            raise ValueError(
                "Provide exactly one of `frames` or `fastf1_session`."
            )

        result = PipelineResult()
        t_start = time.perf_counter()

        with TraceWriter(self.config.trace_dir) as tw:
            # ── Frame source (replay stage) ─────────────────────
            frame_iter = self._build_frame_source(frames, fastf1_session, cache_dir)

            # ── Stage 1: replay → frames ────────────────────────
            collected_frames: list[TelemetryFrame] = []
            for i, frame in enumerate(frame_iter):
                if self.config.max_frames is not None and i >= self.config.max_frames:
                    break
                tw.write("frames", frame)
                collected_frames.append(frame)
            result.frames = collected_frames
            result.counters.frames = len(collected_frames)
            logger.info("[replay] collected %d frames", len(collected_frames))

            if self.config.stop_after == StopAfter.REPLAY:
                return self._finalize(result, tw, t_start, stopped_at="replay")

            # ── Stage 2: frames → snapshots ─────────────────────
            snapshots = self._run_state_engine(collected_frames)
            result.snapshots = snapshots
            result.counters.snapshots = len(snapshots)
            for snap in snapshots:
                tw.write("snapshots", snap)
            logger.info("[state] produced %d snapshots", len(snapshots))

            if self.config.stop_after == StopAfter.SNAPSHOTS:
                return self._finalize(result, tw, t_start, stopped_at="snapshots")

            # ── Stage 3: snapshots → candidates → beats ─────────
            beats, candidates = self._run_scheduler(snapshots, tw, result.counters)
            result.candidates = candidates
            result.beats = beats
            result.counters.candidate_events = len(candidates)
            result.counters.beats_total = len(beats)
            result.counters.beats_selected = sum(
                1 for b in beats if b.selected_for_generation
            )
            result.counters.beats_suppressed = (
                result.counters.beats_total - result.counters.beats_selected
            )
            logger.info(
                "[scheduler] %d candidates → %d beats (%d selected)",
                len(candidates), len(beats), result.counters.beats_selected,
            )

            if self.config.stop_after == StopAfter.BEATS:
                return self._finalize(result, tw, t_start, stopped_at="beats")

            # ── Stage 4: generation (local OR baseline) ─────────
            if self.config.mode == PipelineMode.BASELINE:
                result.baselines = self._run_baseline(snapshots, beats, tw)
                result.counters.baselines = len(result.baselines)
                result.counters.total_gen_latency_ms = sum(
                    r.latency_ms for r in result.baselines
                )
                result.counters.generations = len(result.baselines)
            else:
                result.generated = self._run_local_generator(snapshots, beats, tw)
                result.counters.generations = len(result.generated)
                result.counters.total_gen_latency_ms = sum(
                    r.latency_ms for r in result.generated
                )

            if self.config.stop_after == StopAfter.GENERATION:
                return self._finalize(result, tw, t_start, stopped_at="generation")

            # ── Stage 5: beat manager (local mode only) ─────────
            if (
                self.config.mode == PipelineMode.LOCAL
                and self.config.enable_beat_manager
            ):
                finals = self._run_beat_manager(
                    result.generated, beats, snapshots, tw,
                )
                result.finals = finals
                result.counters.finals_emitted = sum(
                    1 for f in finals if not f.suppression_applied
                )
                result.counters.finals_suppressed = sum(
                    1 for f in finals if f.suppression_applied
                )
                result.counters.fallbacks_used = sum(
                    1 for f in finals
                    if f.suppression_reason.value == "generation_failure"
                )

                # Output adapters (terminal / SSE) — after traces are written.
                self._run_output_adapters(finals)

            return self._finalize(result, tw, t_start, stopped_at=None)

    # ── Internals ──────────────────────────────────────────────────

    def _build_frame_source(
        self,
        frames: Optional[Iterable[TelemetryFrame]],
        fastf1_session: Optional[tuple[int, str, str]],
        cache_dir: Optional[str],
    ) -> Iterable[TelemetryFrame]:
        """Resolve the frame source. Lazy-loads FastF1 so tests never need it."""
        if frames is not None:
            return frames

        assert fastf1_session is not None
        year, gp, session_type = fastf1_session
        from src.telemetry_to_narrative.adapters.session_loader import load_session
        from src.telemetry_to_narrative.adapters.replay_adapter import ReplayAdapter

        session = load_session(year, gp, session_type, cache_dir=cache_dir)
        adapter = ReplayAdapter(session, speed=max(self.config.replay_speed, 0.01))
        return adapter.replay(
            max_frames=self.config.max_frames,
            realtime=self.config.realtime,
        )

    def _run_state_engine(
        self, frames: list[TelemetryFrame],
    ) -> list[RaceStateSnapshot]:
        engine = RaceStateEngine()
        extractor = FeatureExtractor(engine)
        snapshots: list[RaceStateSnapshot] = []
        for frame in frames:
            engine.update(frame)
            snapshots.append(extractor.extract(frame))
        return snapshots

    def _run_scheduler(
        self,
        snapshots: list[RaceStateSnapshot],
        tw: TraceWriter,
        counters: PipelineCounters,
    ) -> tuple[list[ScheduledBeat], list]:
        scheduler = EditorialScheduler()
        scheduler.set_total_laps(max(1, len(snapshots)))

        all_candidates = []
        all_beats: list[ScheduledBeat] = []
        prev_snap: Optional[RaceStateSnapshot] = None

        for snap in snapshots:
            raw_candidates = extract_candidates(snap, prev_snap)
            candidates = _apply_scenario_filter(raw_candidates, self.config.scenario)

            for c in candidates:
                tw.write("candidates", c)
            all_candidates.extend(candidates)

            safety_car = bool(snap.global_state.safety_car)
            beats = scheduler.schedule(
                candidates, frame_id=snap.frame_id, safety_car_active=safety_car,
            )
            for b in beats:
                tw.write("beats", b)
            all_beats.extend(beats)

            prev_snap = snap

        return all_beats, all_candidates

    def _run_local_generator(
        self,
        snapshots: list[RaceStateSnapshot],
        beats: list[ScheduledBeat],
        tw: TraceWriter,
    ) -> list[GeneratedCommentary]:
        backend = load_backend(
            backend_type=self.config.local_backend,
            model_path=self.config.local_model_path,
            adapter_path=self.config.local_adapter_path,
        )
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            backend=backend.model_variant,
        )
        generator = CommentaryGenerator(backend, gen_config)

        results: list[GeneratedCommentary] = []
        try:
            results = generator.generate_batch(
                snapshots, beats, max_items=self.config.max_generation_items,
            )
        except Exception as exc:
            # Graceful degradation: a backend crash should not abort the
            # whole run — the beat manager can still emit fallbacks.
            logger.error("Local generator failed: %s", exc, exc_info=True)

        for r in results:
            tw.write("generated", r)
        return results

    def _run_baseline(
        self,
        snapshots: list[RaceStateSnapshot],
        beats: list[ScheduledBeat],
        tw: TraceWriter,
    ) -> list[BaselineCommentary]:
        backend = load_baseline_backend(
            backend_type=self.config.baseline_backend,
            model_name=self.config.baseline_model_name,
            api_key=self.config.baseline_api_key,
            base_url=self.config.baseline_base_url,
        )
        bl_config = BaselineGenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            provider=backend.provider,
            model_name=backend.model_name,
        )
        generator = BaselineGenerator(backend, bl_config)

        results: list[BaselineCommentary] = []
        try:
            results = generator.generate_batch(
                snapshots, beats, max_items=self.config.max_generation_items,
            )
        except Exception as exc:
            logger.error("Baseline generator failed: %s", exc, exc_info=True)

        for r in results:
            tw.write("baselines", r)
        return results

    def _run_beat_manager(
        self,
        generated: list[GeneratedCommentary],
        beats: list[ScheduledBeat],
        snapshots: list[RaceStateSnapshot],
        tw: TraceWriter,
    ):
        # include_suppressed: suppressed items are already logged at
        # INFO level by BeatManager, so the trace stays inspectable
        # even when they don't reach output adapters.
        manager = BeatManager(
            refresh_interval=self.config.beat_refresh_interval,
            include_suppressed=True,
        )
        finals = manager.process_batch(beats, generated, snapshots)
        for f in finals:
            tw.write("finals", f)
        return finals

    def _run_output_adapters(self, finals) -> None:
        if self.config.enable_terminal_output:
            term = TerminalAdapter(
                show_suppressed=self.config.include_suppressed_in_output,
            )
            term.emit_batch(finals)
            term.print_summary()

        if self.config.enable_sse_output:
            sse = SSEAdapter(
                include_suppressed=self.config.include_suppressed_in_output,
            )
            for f in finals:
                sse.emit(f)
            sys.stdout.write("event: done\ndata: {}\n\n")
            sys.stdout.flush()

    # ── Finalization ────────────────────────────────────────────────

    def _finalize(
        self,
        result: PipelineResult,
        tw: TraceWriter,
        t_start: float,
        stopped_at: Optional[str],
    ) -> PipelineResult:
        result.stopped_at = stopped_at
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        summary = {
            "mode": self.config.mode.value,
            "stopped_at": stopped_at or "complete",
            "scenario": self.config.scenario.value,
            "session_label": self.config.session_label,
            "elapsed_ms": round(elapsed_ms, 2),
            "counters": result.counters.as_dict(),
            "trace_paths": tw.paths,
            "flags": {
                "enable_retrieval": self.config.enable_retrieval,
                "enable_grounding_guard": self.config.enable_grounding_guard,
                "enable_beat_manager": self.config.enable_beat_manager,
                "enable_terminal_output": self.config.enable_terminal_output,
                "enable_sse_output": self.config.enable_sse_output,
            },
            "backend": {
                "local": self.config.local_backend,
                "baseline": self.config.baseline_backend,
                "baseline_model": self.config.baseline_model_name,
            },
        }
        tw.write_summary(summary)
        result.trace_paths = tw.paths

        logger.info(
            "Pipeline run complete (%s, stopped_at=%s) in %.1fms — %s",
            self.config.mode.value, stopped_at or "end", elapsed_ms,
            result.counters.as_dict(),
        )
        return result
