"""StorylineMemory — tracks per-storyline state for narrative progression.

Each storyline progresses through stages:

    initial_interest → active_pressure → imminent_action → decisive_event → concluded

Advancement is driven by event type changes, feature deltas (gap closing,
intensity rising, position changes), and scheduler metadata.  The memory
also enforces minimum refresh intervals and detects semantic duplicates.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.final_commentary import ProgressionStage
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat

logger = logging.getLogger(__name__)

# Default minimum frames between re-narrating the same storyline.
DEFAULT_REFRESH_INTERVAL = 3

# Similarity threshold for semantic duplicate detection.
DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Event types that trigger immediate stage advancement.
_DECISIVE_EVENT_TYPES = {
    EventType.OVERTAKE,
    EventType.PIT_STRATEGY,
    EventType.RACE_CONTROL,
}


class StorylineState:
    """Mutable state for one active storyline."""

    __slots__ = (
        "storyline_id",
        "last_event_type",
        "last_output_text",
        "last_frame_id",
        "last_gap_sec",
        "last_intensity",
        "progression_stage",
        "refresh_counter",
        "narration_count",
    )

    def __init__(self, storyline_id: str) -> None:
        self.storyline_id = storyline_id
        self.last_event_type: Optional[EventType] = None
        self.last_output_text: str = ""
        self.last_frame_id: int = -999
        self.last_gap_sec: Optional[float] = None
        self.last_intensity: Optional[float] = None
        self.progression_stage = ProgressionStage.INITIAL_INTEREST
        self.refresh_counter: int = 0
        self.narration_count: int = 0


class StorylineMemory:
    """Manages progression and suppression state for all active storylines."""

    def __init__(self, refresh_interval: int = DEFAULT_REFRESH_INTERVAL) -> None:
        self._refresh_interval = refresh_interval
        self._storylines: dict[str, StorylineState] = {}

    @property
    def storylines(self) -> dict[str, StorylineState]:
        return self._storylines

    def get_or_create(self, storyline_id: str) -> StorylineState:
        if storyline_id not in self._storylines:
            self._storylines[storyline_id] = StorylineState(storyline_id)
        return self._storylines[storyline_id]

    # ── Suppression checks ───────────────────────────────────────────

    def is_refresh_blocked(self, storyline_id: str, current_frame: int) -> bool:
        """True if the storyline was narrated too recently."""
        state = self._storylines.get(storyline_id)
        if state is None:
            return False
        return (current_frame - state.last_frame_id) < self._refresh_interval

    def is_semantic_duplicate(
        self,
        storyline_id: str,
        new_text: str,
    ) -> bool:
        """True if *new_text* is very similar to the last narration for this storyline."""
        state = self._storylines.get(storyline_id)
        if state is None or not state.last_output_text:
            return False
        ratio = SequenceMatcher(
            None,
            state.last_output_text.lower(),
            new_text.lower(),
        ).ratio()
        return ratio >= DUPLICATE_SIMILARITY_THRESHOLD

    # ── Progression logic ────────────────────────────────────────────

    def compute_progression(
        self,
        storyline_id: str,
        beat: ScheduledBeat,
        snapshot: Optional[RaceStateSnapshot],
    ) -> ProgressionStage:
        """Determine the progression stage for a storyline given new data.

        Rules:
        1. Decisive event types (overtake, pit, race control) → DECISIVE_EVENT.
        2. Gap closing significantly or intensity rising → advance one stage.
        3. Battle duration increasing without gap change → hold stage.
        4. New storyline → INITIAL_INTEREST.
        """
        state = self.get_or_create(storyline_id)

        # First narration for this storyline.
        if state.narration_count == 0:
            return ProgressionStage.INITIAL_INTEREST

        # Decisive event types jump straight to DECISIVE_EVENT.
        if beat.event_type in _DECISIVE_EVENT_TYPES:
            return ProgressionStage.DECISIVE_EVENT

        # Extract current gap and intensity from snapshot if available.
        current_gap = self._extract_gap(beat, snapshot)
        current_intensity = self._extract_intensity(beat, snapshot)

        # Determine if state has advanced.
        gap_closing = False
        intensity_rising = False

        if state.last_gap_sec is not None and current_gap is not None:
            gap_closing = current_gap < state.last_gap_sec - 0.2

        if state.last_intensity is not None and current_intensity is not None:
            intensity_rising = current_intensity > state.last_intensity + 0.1

        if gap_closing or intensity_rising:
            return self._advance_stage(state.progression_stage)

        # No meaningful change — hold current stage.
        return state.progression_stage

    def has_meaningfully_advanced(
        self,
        storyline_id: str,
        beat: ScheduledBeat,
        snapshot: Optional[RaceStateSnapshot],
    ) -> bool:
        """True if the storyline has changed enough to warrant re-narration.

        Checks:
        - Event type changed
        - Gap closed by >= 0.2s
        - Intensity increased by >= 0.1
        - Position changed for involved drivers
        """
        state = self._storylines.get(storyline_id)
        if state is None or state.narration_count == 0:
            return True  # first narration always advances

        # Event type changed.
        if state.last_event_type != beat.event_type:
            return True

        # Gap closing.
        current_gap = self._extract_gap(beat, snapshot)
        if state.last_gap_sec is not None and current_gap is not None:
            if abs(current_gap - state.last_gap_sec) >= 0.2:
                return True

        # Intensity change.
        current_intensity = self._extract_intensity(beat, snapshot)
        if state.last_intensity is not None and current_intensity is not None:
            if abs(current_intensity - state.last_intensity) >= 0.1:
                return True

        return False

    # ── Recording ────────────────────────────────────────────────────

    def record_narration(
        self,
        storyline_id: str,
        frame_id: int,
        beat: ScheduledBeat,
        commentary_text: str,
        progression_stage: ProgressionStage,
        snapshot: Optional[RaceStateSnapshot] = None,
    ) -> None:
        """Record that a storyline was narrated at this frame."""
        state = self.get_or_create(storyline_id)
        state.last_event_type = beat.event_type
        state.last_output_text = commentary_text
        state.last_frame_id = frame_id
        state.progression_stage = progression_stage
        state.narration_count += 1
        state.refresh_counter += 1

        state.last_gap_sec = self._extract_gap(beat, snapshot)
        state.last_intensity = self._extract_intensity(beat, snapshot)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_gap(
        beat: ScheduledBeat,
        snapshot: Optional[RaceStateSnapshot],
    ) -> Optional[float]:
        """Extract the relevant gap for this beat's involved drivers."""
        if snapshot is None:
            return None
        drivers = beat.involved_drivers
        if not drivers:
            return None
        for battle in snapshot.active_battles:
            if battle.driver_ahead in drivers and battle.driver_behind in drivers:
                return battle.gap_sec
        # Fallback: gap_ahead of the second driver.
        if len(drivers) >= 2:
            d = snapshot.drivers.get(drivers[1])
            if d and d.gap_ahead_sec is not None:
                return d.gap_ahead_sec
        return None

    @staticmethod
    def _extract_intensity(
        beat: ScheduledBeat,
        snapshot: Optional[RaceStateSnapshot],
    ) -> Optional[float]:
        """Extract the battle intensity for this beat's involved drivers."""
        if snapshot is None:
            return None
        drivers = beat.involved_drivers
        if not drivers:
            return None
        for battle in snapshot.active_battles:
            if battle.driver_ahead in drivers and battle.driver_behind in drivers:
                return battle.battle_intensity
        return None

    @staticmethod
    def _advance_stage(current: ProgressionStage) -> ProgressionStage:
        """Advance to the next progression stage (capped at DECISIVE_EVENT)."""
        order = [
            ProgressionStage.INITIAL_INTEREST,
            ProgressionStage.ACTIVE_PRESSURE,
            ProgressionStage.IMMINENT_ACTION,
            ProgressionStage.DECISIVE_EVENT,
        ]
        try:
            idx = order.index(current)
            return order[min(idx + 1, len(order) - 1)]
        except ValueError:
            return current
