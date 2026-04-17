"""BeatManager — turns generated commentary into coherent storyline output.

Responsibilities
================
1. Group beats by storyline_id.
2. Track progression stages per storyline.
3. Suppress repetitive or semantically duplicate commentary.
4. Enforce minimum refresh intervals.
5. Build compact race-state summaries.
6. Emit FinalCommentary objects for non-suppressed beats.
7. Provide fallback text for generation failures or uncertain state.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.final_commentary import (
    ConfidenceBand,
    FinalCommentary,
    FinalSuppressionReason,
    ProgressionStage,
    RaceStateSummary,
)
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.beat_manager.storyline_memory import StorylineMemory

logger = logging.getLogger(__name__)

# Fallback text when generation failed or is unavailable.
_FALLBACK_TEXTS: dict[str, str] = {
    "lead_battle": "The battle for the lead continues.",
    "podium_battle": "The fight for the podium is heating up.",
    "midfield_battle": "Close racing in the midfield pack.",
    "overtake": "A position change on track.",
    "pit_strategy": "A strategic pit stop in progress.",
    "race_control": "Race control has intervened.",
    "telemetry_change": "The race continues.",
}
_DEFAULT_FALLBACK = "Racing continues."


def _make_final_id() -> str:
    return uuid.uuid4().hex[:16]


def build_race_state_summary(
    snapshot: Optional[RaceStateSnapshot],
    involved_drivers: list[str],
) -> RaceStateSummary:
    """Build a compact race-state summary for FinalCommentary."""
    if snapshot is None:
        return RaceStateSummary()

    # Lap from first available driver.
    lap = None
    for d in snapshot.drivers.values():
        if d.lap_number is not None:
            lap = d.lap_number
            break

    # Per-driver summary.
    driver_summary: dict[str, dict] = {}
    for abbr in involved_drivers:
        ds = snapshot.drivers.get(abbr)
        if ds is None:
            continue
        driver_summary[abbr] = {
            "position": ds.position,
            "gap_ahead_sec": ds.gap_ahead_sec,
            "tire_compound": ds.tire_compound,
            "tire_age_laps": ds.tire_age_laps,
        }

    # Highest battle intensity among involved drivers.
    max_intensity: Optional[float] = None
    involved_set = set(involved_drivers)
    for b in snapshot.active_battles:
        if b.driver_ahead in involved_set or b.driver_behind in involved_set:
            if b.battle_intensity is not None:
                if max_intensity is None or b.battle_intensity > max_intensity:
                    max_intensity = b.battle_intensity

    return RaceStateSummary(
        lap=lap,
        leader=snapshot.derived_summary.leader,
        flag_state=snapshot.global_state.track_status,
        safety_car=bool(snapshot.global_state.safety_car),
        involved_driver_summary=driver_summary,
        battle_priority=max_intensity,
    )


def _compute_confidence(
    progression: ProgressionStage,
    commentary_text: str,
    is_fallback: bool,
) -> ConfidenceBand:
    """Determine confidence band from progression state and output quality."""
    if is_fallback:
        return ConfidenceBand.LOW
    if progression == ProgressionStage.DECISIVE_EVENT:
        return ConfidenceBand.HIGH
    if progression in (ProgressionStage.ACTIVE_PRESSURE, ProgressionStage.IMMINENT_ACTION):
        return ConfidenceBand.HIGH if len(commentary_text) > 30 else ConfidenceBand.MEDIUM
    return ConfidenceBand.MEDIUM


class BeatManager:
    """Post-processes generated commentary into coherent storyline output.

    Usage::

        manager = BeatManager()
        finals = manager.process_batch(snapshots, beats, commentaries)
    """

    def __init__(
        self,
        refresh_interval: int = 3,
        include_suppressed: bool = False,
    ) -> None:
        self.memory = StorylineMemory(refresh_interval=refresh_interval)
        self._include_suppressed = include_suppressed

    def process(
        self,
        beat: ScheduledBeat,
        commentary: Optional[GeneratedCommentary],
        snapshot: Optional[RaceStateSnapshot] = None,
    ) -> Optional[FinalCommentary]:
        """Process a single beat + commentary into a FinalCommentary.

        Returns None if the commentary is suppressed and
        ``include_suppressed`` is False.
        """
        storyline_id = beat.storyline_id or f"unnamed_{beat.beat_id}"
        frame_id = beat.source_frame_id

        # Handle generation failure.
        if commentary is None or not commentary.commentary_text.strip():
            return self._emit_fallback(
                beat, storyline_id, frame_id, snapshot,
                FinalSuppressionReason.GENERATION_FAILURE,
            )

        text = commentary.commentary_text

        # 1. Refresh interval check.
        if self.memory.is_refresh_blocked(storyline_id, frame_id):
            return self._emit_suppressed(
                beat, commentary, storyline_id, snapshot,
                FinalSuppressionReason.REFRESH_INTERVAL,
            )

        # 2. Semantic duplicate check.
        if self.memory.is_semantic_duplicate(storyline_id, text):
            return self._emit_suppressed(
                beat, commentary, storyline_id, snapshot,
                FinalSuppressionReason.DUPLICATE_SEMANTICS,
            )

        # 3. Meaningful advancement check.
        if not self.memory.has_meaningfully_advanced(storyline_id, beat, snapshot):
            return self._emit_suppressed(
                beat, commentary, storyline_id, snapshot,
                FinalSuppressionReason.NO_PROGRESSION,
            )

        # Passed all checks — compute progression and emit.
        progression = self.memory.compute_progression(storyline_id, beat, snapshot)
        confidence = _compute_confidence(progression, text, is_fallback=False)

        race_summary = build_race_state_summary(snapshot, beat.involved_drivers)
        replay_lag = self._compute_replay_lag(beat, commentary)

        final = FinalCommentary(
            final_id=_make_final_id(),
            timestamp=commentary.timestamp,
            beat_id=beat.beat_id,
            storyline_id=storyline_id,
            event_type=beat.event_type,
            involved_drivers=beat.involved_drivers,
            final_text=text,
            confidence_band=confidence,
            replay_lag_seconds=replay_lag,
            race_state_summary=race_summary,
            suppression_applied=False,
            suppression_reason=FinalSuppressionReason.NONE,
            progression_stage=progression,
            source_generation_id=commentary.generation_id,
        )

        # Record in memory.
        self.memory.record_narration(
            storyline_id, frame_id, beat, text, progression, snapshot,
        )

        logger.info(
            "Final commentary [%s] %s stage=%s: %s",
            storyline_id, beat.event_type.value,
            progression.value, text[:60],
        )

        return final

    def process_batch(
        self,
        beats: list[ScheduledBeat],
        commentaries: list[GeneratedCommentary],
        snapshots: Optional[list[RaceStateSnapshot]] = None,
    ) -> list[FinalCommentary]:
        """Process a batch of beats + commentaries.

        Matches commentaries to beats by beat_id.
        """
        # Build lookups.
        commentary_by_beat: dict[str, GeneratedCommentary] = {
            c.beat_id: c for c in commentaries
        }
        snap_by_frame: dict[int, RaceStateSnapshot] = {}
        if snapshots:
            snap_by_frame = {s.frame_id: s for s in snapshots}

        results: list[FinalCommentary] = []
        for beat in beats:
            if not beat.selected_for_generation:
                continue

            commentary = commentary_by_beat.get(beat.beat_id)
            snapshot = snap_by_frame.get(beat.source_frame_id)

            final = self.process(beat, commentary, snapshot)
            if final is not None:
                results.append(final)

        logger.info(
            "Beat manager processed %d beats → %d final commentaries.",
            len(beats), len(results),
        )
        return results

    # ── Internal helpers ─────────────────────────────────────────────

    def _emit_suppressed(
        self,
        beat: ScheduledBeat,
        commentary: GeneratedCommentary,
        storyline_id: str,
        snapshot: Optional[RaceStateSnapshot],
        reason: FinalSuppressionReason,
    ) -> Optional[FinalCommentary]:
        """Emit a suppressed FinalCommentary or None."""
        logger.info(
            "Suppressed [%s] %s: %s",
            storyline_id, reason.value, commentary.commentary_text[:40],
        )
        if not self._include_suppressed:
            return None

        state = self.memory.get_or_create(storyline_id)
        race_summary = build_race_state_summary(snapshot, beat.involved_drivers)

        return FinalCommentary(
            final_id=_make_final_id(),
            timestamp=commentary.timestamp,
            beat_id=beat.beat_id,
            storyline_id=storyline_id,
            event_type=beat.event_type,
            involved_drivers=beat.involved_drivers,
            final_text=commentary.commentary_text,
            confidence_band=ConfidenceBand.SUPPRESSED,
            race_state_summary=race_summary,
            suppression_applied=True,
            suppression_reason=reason,
            progression_stage=state.progression_stage,
            source_generation_id=commentary.generation_id,
        )

    def _emit_fallback(
        self,
        beat: ScheduledBeat,
        storyline_id: str,
        frame_id: int,
        snapshot: Optional[RaceStateSnapshot],
        reason: FinalSuppressionReason,
    ) -> Optional[FinalCommentary]:
        """Emit fallback commentary when generation failed."""
        fallback_text = _FALLBACK_TEXTS.get(beat.event_type.value, _DEFAULT_FALLBACK)
        race_summary = build_race_state_summary(snapshot, beat.involved_drivers)
        state = self.memory.get_or_create(storyline_id)

        final = FinalCommentary(
            final_id=_make_final_id(),
            timestamp=datetime.now(timezone.utc),
            beat_id=beat.beat_id,
            storyline_id=storyline_id,
            event_type=beat.event_type,
            involved_drivers=beat.involved_drivers,
            final_text=fallback_text,
            confidence_band=ConfidenceBand.LOW,
            race_state_summary=race_summary,
            suppression_applied=True,
            suppression_reason=reason,
            progression_stage=state.progression_stage,
            source_generation_id=None,
        )

        logger.info(
            "Fallback commentary [%s] %s: %s",
            storyline_id, reason.value, fallback_text,
        )
        return final

    @staticmethod
    def _compute_replay_lag(
        beat: ScheduledBeat,
        commentary: GeneratedCommentary,
    ) -> float:
        """Compute lag between event time and commentary emission."""
        delta = commentary.timestamp - beat.timestamp
        return max(0.0, delta.total_seconds())
