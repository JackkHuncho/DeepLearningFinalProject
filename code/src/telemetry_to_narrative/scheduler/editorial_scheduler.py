"""EditorialScheduler — converts candidate events into prioritized narrative beats.

Responsibilities
================
1. Score each CandidateEvent using a decomposed, explainable priority.
2. Select the highest-value beat for immediate narration.
3. Suppress, delay, or merge lower-priority beats.
4. Track storylines to avoid redundant narration.
5. Apply backpressure when the downstream generation pipeline is loaded.
6. Expose observability counters for debugging and evaluation.

Scoring formula
===============
    raw = w_ctx  * contextual_importance(event_type)
        + w_urg  * urgency(gap, intensity, safety)
        + w_phase* race_phase(lap_pct, flag_state)
        + w_story* storyline_relevance(storyline_tracker)
        + w_conf * confidence

    priority = max(0, raw - backpressure_penalty)

Each component is documented below in its helper function.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Optional

from src.telemetry_to_narrative.schemas.candidate_event import (
    CandidateEvent,
    EventType,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScoreBreakdown,
    ScheduledBeat,
    SuppressionReason,
)
from src.telemetry_to_narrative.config.scheduler_config import SchedulerConfig

logger = logging.getLogger(__name__)


# ── Observability counters ────────────────────────────────────────────────

class SchedulerCounters:
    """Lightweight counters for monitoring scheduler decisions."""

    __slots__ = (
        "events_received",
        "beats_selected",
        "beats_suppressed",
        "beats_dropped",
        "retrieval_disabled_count",
        "queue_depth_high_water",
    )

    def __init__(self) -> None:
        self.events_received: int = 0
        self.beats_selected: int = 0
        self.beats_suppressed: int = 0
        self.beats_dropped: int = 0
        self.retrieval_disabled_count: int = 0
        self.queue_depth_high_water: int = 0

    def as_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "beats_selected": self.beats_selected,
            "beats_suppressed": self.beats_suppressed,
            "beats_dropped": self.beats_dropped,
            "retrieval_disabled_count": self.retrieval_disabled_count,
            "queue_depth_high_water": self.queue_depth_high_water,
        }


# ── Storyline tracker ────────────────────────────────────────────────────

def _derive_storyline_id(event: CandidateEvent) -> str:
    """Deterministic storyline ID from event type and drivers.

    Battles between the same pair of drivers share a storyline.
    Race-control events share a single "race_control" storyline.
    Pit strategy is per-driver.
    """
    etype = event.event_type.value
    drivers_key = "_".join(sorted(event.involved_drivers)) if event.involved_drivers else "global"

    if etype in ("race_control",):
        return "race_control"
    if etype in ("lead_battle", "podium_battle", "midfield_battle"):
        return f"battle_{drivers_key}"
    if etype in ("pit_strategy",):
        return f"pit_{drivers_key}"
    if etype in ("overtake",):
        return f"overtake_{drivers_key}"
    return f"{etype}_{drivers_key}"


class _StorylineState:
    """Per-storyline mutable state."""

    __slots__ = ("storyline_id", "last_narrated_frame", "last_gap_sec", "narration_count")

    def __init__(self, storyline_id: str) -> None:
        self.storyline_id = storyline_id
        self.last_narrated_frame: int = -999  # never narrated
        self.last_gap_sec: Optional[float] = None
        self.narration_count: int = 0


class StorylineTracker:
    """Tracks active storylines to prevent redundant narration."""

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._storylines: dict[str, _StorylineState] = {}

    def get_or_create(self, storyline_id: str) -> _StorylineState:
        if storyline_id not in self._storylines:
            self._storylines[storyline_id] = _StorylineState(storyline_id)
        return self._storylines[storyline_id]

    def is_on_cooldown(self, storyline_id: str, current_frame: int) -> bool:
        """True if this storyline was narrated too recently."""
        state = self._storylines.get(storyline_id)
        if state is None:
            return False
        return (current_frame - state.last_narrated_frame) < self._config.storyline.cooldown_frames

    def has_advanced(self, storyline_id: str, event: CandidateEvent) -> bool:
        """True if the storyline has meaningfully changed since last narration.

        For battle storylines: gap must have changed by at least
        ``battle_advance_threshold_sec``.

        For overtake storylines: always counts as advanced (position changed).

        For other types: always counts as advanced.
        """
        state = self._storylines.get(storyline_id)
        if state is None:
            return True  # first occurrence is always "advanced"

        etype = event.event_type.value
        if etype in ("lead_battle", "podium_battle", "midfield_battle"):
            if state.last_gap_sec is not None and event.gap_sec is not None:
                delta = abs(event.gap_sec - state.last_gap_sec)
                return delta >= self._config.storyline.battle_advance_threshold_sec
            return True  # missing data → allow

        if etype == "overtake":
            return self._config.storyline.require_position_change_for_overtake

        return True

    def record_narration(self, storyline_id: str, frame_id: int, gap_sec: Optional[float]) -> None:
        """Mark a storyline as narrated at this frame."""
        state = self.get_or_create(storyline_id)
        state.last_narrated_frame = frame_id
        state.last_gap_sec = gap_sec
        state.narration_count += 1


# ── Scoring helpers ───────────────────────────────────────────────────────

def score_contextual_importance(event: CandidateEvent, config: SchedulerConfig) -> float:
    """Map event type to a base importance score via the precedence tiers.

    Higher tiers (safety-critical, lead battles) score closer to 1.0.
    Unknown event types default to 0.1.
    """
    return config.precedence_tiers.get(event.event_type.value, 0.1)


def score_urgency(event: CandidateEvent) -> float:
    """Score urgency in [0, 1] based on gap tightness, intensity, and safety.

    Formula:
        gap_urgency  = clamp(1 - gap / 2.0, 0, 1)   if gap available, else 0.3
        intensity    = event.battle_intensity          if available, else 0.0
        safety_boost = 1.0 if safety_critical else 0.0

        urgency = 0.40 * gap_urgency + 0.35 * intensity + 0.25 * safety_boost
    """
    if event.safety_critical:
        return 1.0  # always maximally urgent

    gap_urgency = 0.3  # default when gap unknown
    if event.gap_sec is not None:
        gap_urgency = max(0.0, min(1.0, 1.0 - event.gap_sec / 2.0))

    intensity = event.battle_intensity if event.battle_intensity is not None else 0.0

    return 0.40 * gap_urgency + 0.35 * intensity + 0.25 * 0.0


def score_race_phase(
    lap_number: Optional[int],
    total_laps: int,
    safety_car_active: bool,
) -> float:
    """Boost events that occur during critical race phases.

    Critical phases (score 1.0):
    - First 3 laps (opening chaos)
    - Last 10% of race (closing drama)
    - Safety-car restart windows

    Mid-race steady state scores 0.3.

    Formula:
        if lap <= 3: 0.9
        elif lap > 0.9 * total: 0.8
        elif safety_car_active: 0.7
        else: 0.3
    """
    if lap_number is None or total_laps <= 0:
        return 0.3

    if lap_number <= 3:
        return 0.9
    if total_laps > 0 and lap_number > 0.9 * total_laps:
        return 0.8
    if safety_car_active:
        return 0.7
    return 0.3


def score_storyline_relevance(
    tracker: StorylineTracker,
    storyline_id: str,
    event: CandidateEvent,
    current_frame: int,
) -> float:
    """Score how relevant this event is to an ongoing storyline.

    An event that *continues* a known, active storyline scores higher
    because narrative threads build engagement.

    Formula:
        if new storyline: 0.6  (novelty bonus)
        elif advanced and not on cooldown: 0.8  (continuation bonus)
        elif on cooldown: 0.1
        elif not advanced: 0.1
    """
    state = tracker._storylines.get(storyline_id)

    if state is None or state.narration_count == 0:
        return 0.6  # new storyline — novelty

    if tracker.is_on_cooldown(storyline_id, current_frame):
        return 0.1

    if tracker.has_advanced(storyline_id, event):
        return 0.8  # continuation of an evolving story

    return 0.1  # stale — no meaningful change


def apply_backpressure_adjustments(
    config: SchedulerConfig,
    queue_depth: int,
    latency_estimate_ms: float,
) -> tuple[float, bool, int]:
    """Compute backpressure penalty, retrieval flag, and length budget.

    Returns
    -------
    penalty : float
        Score penalty to subtract.
    retrieval_allowed : bool
        Whether retrieval should be used for this beat.
    length_budget : int
        Target word count.
    """
    bp = config.backpressure
    penalty = 0.0
    retrieval_allowed = True
    length_budget = config.default_length_budget

    # Penalty scales with queue depth above soft limit.
    if queue_depth > bp.queue_soft_limit:
        excess = queue_depth - bp.queue_soft_limit
        penalty = excess * bp.penalty_per_item

    # Disable retrieval if queue is very deep or latency is high.
    if queue_depth >= bp.queue_retrieval_cutoff or latency_estimate_ms >= bp.latency_retrieval_cutoff_ms:
        retrieval_allowed = False

    # Shorten output budget under load.
    if queue_depth >= bp.queue_hard_limit:
        length_budget = bp.minimal_length_budget
    elif queue_depth >= bp.queue_soft_limit:
        length_budget = bp.reduced_length_budget

    return penalty, retrieval_allowed, length_budget


# ── Main scheduler ────────────────────────────────────────────────────────

class EditorialScheduler:
    """Converts candidate events into prioritized ScheduledBeats.

    Usage::

        scheduler = EditorialScheduler()
        beats = scheduler.schedule(candidates, frame_id=5, queue_depth=0)

    The scheduler is stateful: it tracks storylines across calls so that
    repeated events are suppressed appropriately.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self.storylines = StorylineTracker(self.config)
        self.counters = SchedulerCounters()

        # Simulated queue depth and latency — set externally or via CLI.
        self._queue_depth: int = 0
        self._latency_estimate_ms: float = 0.0

        # Approximate total laps for race-phase scoring.
        self._total_laps: int = 57  # default (Monza-ish)

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @queue_depth.setter
    def queue_depth(self, value: int) -> None:
        self._queue_depth = max(0, value)
        self.counters.queue_depth_high_water = max(
            self.counters.queue_depth_high_water, self._queue_depth
        )

    @property
    def latency_estimate_ms(self) -> float:
        return self._latency_estimate_ms

    @latency_estimate_ms.setter
    def latency_estimate_ms(self, value: float) -> None:
        self._latency_estimate_ms = max(0.0, value)

    def set_total_laps(self, laps: int) -> None:
        self._total_laps = max(1, laps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        candidates: list[CandidateEvent],
        frame_id: int,
        safety_car_active: bool = False,
    ) -> list[ScheduledBeat]:
        """Score, rank, and filter a batch of candidate events.

        Returns all beats (selected *and* suppressed) so the caller can
        inspect decisions.  Only beats with ``selected_for_generation=True``
        should be passed to the generation pipeline.
        """
        self.counters.events_received += len(candidates)

        # Phase 1: score every candidate.
        scored: list[tuple[float, ScheduledBeat]] = []
        for event in candidates:
            beat = self._score_event(event, frame_id, safety_car_active)
            scored.append((beat.priority_score, beat))

        # Phase 2: sort by priority (descending).
        scored.sort(key=lambda t: t[0], reverse=True)

        # Phase 3: select the top beat, suppress the rest.
        beats: list[ScheduledBeat] = []
        selected_one = False
        for _, beat in scored:
            if not selected_one and beat.selected_for_generation:
                # Accept this beat.
                selected_one = True
                self.counters.beats_selected += 1
                # Record narration in the storyline tracker.
                if beat.storyline_id:
                    gap_sec = None
                    for c in candidates:
                        if c.event_id == beat.beat_id:
                            gap_sec = c.gap_sec
                            break
                    self.storylines.record_narration(beat.storyline_id, frame_id, gap_sec)
                beats.append(beat)
            else:
                # Suppress or downgrade.
                if beat.selected_for_generation:
                    # Was eligible but we already selected one.
                    beat.selected_for_generation = False
                    beat.suppression_reason = SuppressionReason.LOW_PRIORITY
                self.counters.beats_suppressed += 1
                beats.append(beat)

        # Decrement simulated queue depth for selected beat (it will be consumed).
        if selected_one and self._queue_depth > 0:
            self._queue_depth -= 1

        return beats

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_event(
        self,
        event: CandidateEvent,
        frame_id: int,
        safety_car_active: bool,
    ) -> ScheduledBeat:
        """Score a single candidate and return a ScheduledBeat."""
        storyline_id = _derive_storyline_id(event)
        w = self.config.weights

        # Component scores.
        ctx = score_contextual_importance(event, self.config)
        urg = score_urgency(event)
        phase = score_race_phase(
            event.position_of_interest,
            self._total_laps,
            safety_car_active,
        )
        story = score_storyline_relevance(
            self.storylines, storyline_id, event, frame_id,
        )
        conf = event.confidence

        # Backpressure adjustments.
        bp_penalty, retrieval_allowed, length_budget = apply_backpressure_adjustments(
            self.config, self._queue_depth, self._latency_estimate_ms,
        )

        if not retrieval_allowed:
            self.counters.retrieval_disabled_count += 1

        # Weighted sum.
        raw = (
            w.contextual_importance * ctx
            + w.urgency * urg
            + w.race_phase * phase
            + w.storyline_relevance * story
            + w.confidence * conf
        )
        priority = max(0.0, raw - bp_penalty)

        breakdown = ScoreBreakdown(
            contextual_importance=round(ctx, 4),
            urgency=round(urg, 4),
            race_phase=round(phase, 4),
            storyline_relevance=round(story, 4),
            confidence=round(conf, 4),
            backpressure_penalty=round(bp_penalty, 4),
        )

        # Determine selection.
        selected = True
        suppression = SuppressionReason.NONE

        # Hard drop under extreme backpressure for non-safety events.
        if (
            self._queue_depth >= self.config.backpressure.queue_hard_limit
            and not event.safety_critical
        ):
            selected = False
            suppression = SuppressionReason.BACKPRESSURE_DROP
            self.counters.beats_dropped += 1

        # Threshold check.
        elif priority < self.config.narration_threshold:
            selected = False
            suppression = SuppressionReason.LOW_PRIORITY

        # Cooldown check.
        elif self.storylines.is_on_cooldown(storyline_id, frame_id):
            selected = False
            suppression = SuppressionReason.COOLDOWN

        # Storyline-not-advanced check.
        elif not self.storylines.has_advanced(storyline_id, event):
            selected = False
            suppression = SuppressionReason.STORYLINE_NOT_ADVANCED

        return ScheduledBeat(
            beat_id=event.event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            involved_drivers=event.involved_drivers,
            storyline_id=storyline_id,
            priority_score=round(priority, 4),
            score_breakdown=breakdown,
            selected_for_generation=selected,
            suppression_reason=suppression,
            retrieval_allowed=retrieval_allowed,
            output_length_budget=length_budget,
            source_frame_id=event.source_frame_id,
            description=event.description,
        )
