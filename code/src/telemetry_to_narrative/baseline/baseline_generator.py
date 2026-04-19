"""BaselineGenerator — runs the frontier LLM baseline over scheduled beats.

This is the counterpart to ``CommentaryGenerator`` but without the
local system's structured controls (beat manager, grounding guard,
retrieval gating, suppression/progression).  It gives the baseline
exactly the same structured race-state input and asks for one short
commentary line.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.baseline_commentary import (
    BaselineCommentary,
    BaselineGenerationConfig,
)
from src.telemetry_to_narrative.baseline.backends import BaseBaselineBackend
from src.telemetry_to_narrative.baseline.prompt_builder import build_baseline_prompt

logger = logging.getLogger(__name__)


def _make_baseline_id() -> str:
    return "bl_" + uuid.uuid4().hex[:14]


def _clean_output(raw: str) -> str:
    """Minimal cleanup only — strip whitespace and wrapping quotes.

    The baseline intentionally avoids any structured post-processing
    (no grounding correction, no progression-aware trimming) so that
    the A/B comparison attributes quality differences to system design.
    """
    text = raw.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    return text


class BaselineGenerator:
    """Orchestrates baseline LLM generation over scheduled beats.

    Usage::

        backend = MockBaselineBackend()
        generator = BaselineGenerator(backend)
        baseline = generator.generate(snapshot, beat)
    """

    def __init__(
        self,
        backend: BaseBaselineBackend,
        config: Optional[BaselineGenerationConfig] = None,
    ) -> None:
        self._backend = backend
        self._config = config or BaselineGenerationConfig(
            provider=backend.provider,
            model_name=backend.model_name,
        )

    @property
    def config(self) -> BaselineGenerationConfig:
        return self._config

    @config.setter
    def config(self, value: BaselineGenerationConfig) -> None:
        self._config = value

    def generate(
        self,
        snapshot: RaceStateSnapshot,
        beat: ScheduledBeat,
    ) -> BaselineCommentary:
        """Generate one baseline commentary for a single beat."""
        prompt_text = build_baseline_prompt(snapshot, beat)

        gen_config = BaselineGenerationConfig(
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            provider=self._backend.provider,
            model_name=self._backend.model_name,
        )

        logger.debug("Baseline prompt for beat %s:\n%s", beat.beat_id, prompt_text)

        t0 = time.perf_counter()
        raw_output = self._backend.generate(prompt_text, gen_config)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        commentary_text = _clean_output(raw_output)

        result = BaselineCommentary(
            baseline_id=_make_baseline_id(),
            timestamp=datetime.now(timezone.utc),
            beat_id=beat.beat_id,
            storyline_id=beat.storyline_id,
            event_type=beat.event_type,
            involved_drivers=beat.involved_drivers,
            model_name=self._backend.model_name,
            provider=self._backend.provider,
            prompt_text=prompt_text,
            raw_output_text=raw_output,
            commentary_text=commentary_text,
            output_length=len(raw_output),
            latency_ms=round(latency_ms, 2),
            generation_config=gen_config,
        )

        logger.info(
            "Baseline commentary for beat %s [%s]: %d chars in %.1fms — %s",
            beat.beat_id, self._backend.model_name,
            result.output_length, latency_ms, commentary_text[:80],
        )

        return result

    def generate_batch(
        self,
        snapshots: list[RaceStateSnapshot],
        beats: list[ScheduledBeat],
        max_items: Optional[int] = None,
    ) -> list[BaselineCommentary]:
        """Generate baseline commentary for multiple beats.

        Only selected beats are processed, mirroring the local system's
        event-selection inputs so the two systems cover identical events.
        """
        snap_index: dict[int, RaceStateSnapshot] = {s.frame_id: s for s in snapshots}

        target_beats = [b for b in beats if b.selected_for_generation]
        if max_items is not None:
            target_beats = target_beats[:max_items]

        results: list[BaselineCommentary] = []
        for beat in target_beats:
            snapshot = snap_index.get(beat.source_frame_id)
            if snapshot is None:
                logger.warning(
                    "Beat %s references frame %d not in snapshot list; skipping.",
                    beat.beat_id, beat.source_frame_id,
                )
                continue
            results.append(self.generate(snapshot, beat))

        logger.info(
            "Baseline batch complete: %d / %d beats processed.",
            len(results), len(target_beats),
        )
        return results
