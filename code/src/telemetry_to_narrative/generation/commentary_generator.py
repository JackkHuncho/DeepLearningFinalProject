"""CommentaryGenerator — runtime orchestrator for commentary generation.

Consumes a RaceStateSnapshot and ScheduledBeat, builds a prompt,
runs inference via the configured backend, and returns a
GeneratedCommentary object with full provenance.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary,
    GenerationConfig,
)
from src.telemetry_to_narrative.generation.backends import BaseGeneratorBackend
from src.telemetry_to_narrative.generation.prompt_builder import build_prompt

logger = logging.getLogger(__name__)


def _make_generation_id() -> str:
    return uuid.uuid4().hex[:16]


def _clean_output(raw: str) -> str:
    """Post-process raw model output into clean commentary.

    Strips whitespace, removes any leading/trailing quotes, and
    truncates to the first complete sentence if multiple are present.
    """
    text = raw.strip()
    # Remove wrapping quotes.
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    return text


class CommentaryGenerator:
    """Generates commentary from race state and scheduled beats.

    Usage::

        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        commentary = generator.generate(snapshot, beat)
    """

    def __init__(
        self,
        backend: BaseGeneratorBackend,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self._backend = backend
        self._config = config or GenerationConfig(backend=backend.model_variant)

    @property
    def config(self) -> GenerationConfig:
        return self._config

    @config.setter
    def config(self, value: GenerationConfig) -> None:
        self._config = value

    def generate(
        self,
        snapshot: RaceStateSnapshot,
        beat: ScheduledBeat,
        retrieved_context: Optional[list[str]] = None,
        output_length_budget: Optional[int] = None,
    ) -> GeneratedCommentary:
        """Generate commentary for a single beat.

        Parameters
        ----------
        snapshot : RaceStateSnapshot
            Current race state at the beat's frame.
        beat : ScheduledBeat
            The beat to generate commentary for.
        retrieved_context : list[str] | None
            Optional retrieval-augmented context.
        output_length_budget : int | None
            Override max_new_tokens for this generation.
            If None, uses beat.output_length_budget as a guide.

        Returns
        -------
        GeneratedCommentary
            Full provenance object.
        """
        # Determine if we should use short budget.
        budget = output_length_budget or beat.output_length_budget
        short_budget = budget < 100

        # Build prompt (reuses SFT format).
        prompt_text, retrieval_used = build_prompt(
            snapshot, beat,
            retrieved_context=retrieved_context,
            short_budget=short_budget,
        )

        # Adjust config for this generation.
        gen_config = GenerationConfig(
            max_new_tokens=min(budget, self._config.max_new_tokens),
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
            backend=self._backend.model_variant,
        )

        # Log the prompt.
        logger.debug("Prompt for beat %s:\n%s", beat.beat_id, prompt_text)

        # Run inference with timing.
        t0 = time.perf_counter()
        raw_output = self._backend.generate(prompt_text, gen_config)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        commentary_text = _clean_output(raw_output)

        result = GeneratedCommentary(
            generation_id=_make_generation_id(),
            timestamp=datetime.now(timezone.utc),
            beat_id=beat.beat_id,
            storyline_id=beat.storyline_id,
            event_type=beat.event_type,
            involved_drivers=beat.involved_drivers,
            model_variant=self._backend.model_variant,
            prompt_text=prompt_text,
            raw_output_text=raw_output,
            commentary_text=commentary_text,
            output_length=len(raw_output),
            latency_ms=round(latency_ms, 2),
            retrieved_context_used=retrieval_used,
            generation_config=gen_config,
        )

        logger.info(
            "Generated commentary for beat %s: %d chars in %.1fms — %s",
            beat.beat_id, result.output_length, latency_ms,
            commentary_text[:80],
        )

        return result

    def generate_batch(
        self,
        snapshots: list[RaceStateSnapshot],
        beats: list[ScheduledBeat],
        retrieved_contexts: Optional[dict[str, list[str]]] = None,
        max_items: Optional[int] = None,
    ) -> list[GeneratedCommentary]:
        """Generate commentary for multiple beats.

        Parameters
        ----------
        snapshots : list[RaceStateSnapshot]
            Ordered snapshots (by frame_id).
        beats : list[ScheduledBeat]
            Beats to generate for (typically only selected ones).
        retrieved_contexts : dict[str, list[str]] | None
            Optional mapping of beat_id → context strings.
        max_items : int | None
            Limit the number of generations.

        Returns
        -------
        list[GeneratedCommentary]
        """
        # Build frame_id → snapshot lookup.
        snap_index: dict[int, RaceStateSnapshot] = {s.frame_id: s for s in snapshots}

        # Filter to selected beats only.
        target_beats = [b for b in beats if b.selected_for_generation]
        if max_items is not None:
            target_beats = target_beats[:max_items]

        results: list[GeneratedCommentary] = []
        for beat in target_beats:
            snapshot = snap_index.get(beat.source_frame_id)
            if snapshot is None:
                logger.warning(
                    "Beat %s references frame %d which is not in snapshot list; skipping.",
                    beat.beat_id, beat.source_frame_id,
                )
                continue

            ctx = None
            if retrieved_contexts and beat.beat_id in retrieved_contexts:
                ctx = retrieved_contexts[beat.beat_id]

            result = self.generate(snapshot, beat, retrieved_context=ctx)
            results.append(result)

        logger.info(
            "Batch generation complete: %d / %d beats processed.",
            len(results), len(target_beats),
        )
        return results
