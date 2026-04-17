"""Configurable settings for the EditorialScheduler.

All weights, thresholds, and budget parameters live here so they can be
tuned without touching the scheduler logic itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoringWeights(BaseModel):
    """Weights for the composite priority score.

    The final score is:
        w_ctx * contextual_importance
      + w_urg * urgency
      + w_phase * race_phase
      + w_story * storyline_relevance
      + w_conf * confidence
      - backpressure_penalty

    All weights should sum to roughly 1.0 for interpretability, but this
    is not enforced.
    """

    contextual_importance: float = 0.30
    urgency: float = 0.25
    race_phase: float = 0.10
    storyline_relevance: float = 0.20
    confidence: float = 0.15


class BackpressureConfig(BaseModel):
    """Thresholds that control load-shedding behaviour."""

    # Queue depth at which backpressure starts applying a penalty.
    queue_soft_limit: int = Field(
        default=3,
        description="Queue depth above which a penalty is added to scores",
    )
    # Queue depth at which retrieval is disabled entirely.
    queue_retrieval_cutoff: int = Field(
        default=5,
        description="Queue depth above which retrieval_allowed is set to False",
    )
    # Queue depth at which non-critical events are dropped outright.
    queue_hard_limit: int = Field(
        default=8,
        description="Queue depth above which non-safety-critical events are dropped",
    )
    # Penalty per queue item above the soft limit (subtracted from score).
    penalty_per_item: float = Field(
        default=0.05,
        description="Score penalty for each item above queue_soft_limit",
    )
    # Latency threshold (ms) above which retrieval is disabled.
    latency_retrieval_cutoff_ms: float = Field(
        default=500.0,
        description="Estimated first-beat latency above which retrieval is disabled",
    )
    # Output length reductions under load.
    reduced_length_budget: int = Field(
        default=80,
        description="Word budget when system is under moderate load",
    )
    minimal_length_budget: int = Field(
        default=40,
        description="Word budget when system is under heavy load",
    )


class StorylineConfig(BaseModel):
    """Controls for storyline tracking and duplicate suppression."""

    # Minimum frames between narrations of the same storyline.
    cooldown_frames: int = Field(
        default=3,
        description="Frames before the same storyline can be narrated again",
    )
    # A storyline must show this much gap-change (seconds) to count as
    # "meaningfully advanced" for a battle storyline.
    battle_advance_threshold_sec: float = Field(
        default=0.3,
        description="Minimum gap change to consider a battle storyline advanced",
    )
    # A storyline must show a position change to count as advanced for
    # overtake storylines.
    require_position_change_for_overtake: bool = True


class SchedulerConfig(BaseModel):
    """Top-level scheduler configuration."""

    # Minimum priority score for a beat to be selected for generation.
    narration_threshold: float = Field(
        default=0.35,
        description="Beats scoring below this are suppressed as low_priority",
    )
    # Default output word budget when system is healthy.
    default_length_budget: int = Field(
        default=150,
        description="Target word count for generated commentary",
    )
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    backpressure: BackpressureConfig = Field(default_factory=BackpressureConfig)
    storyline: StorylineConfig = Field(default_factory=StorylineConfig)

    # Editorial precedence tiers — maps EventType.value to a base
    # contextual-importance score in [0, 1].
    precedence_tiers: dict[str, float] = Field(
        default_factory=lambda: {
            "race_control": 1.00,       # tier 1: safety-critical
            "lead_battle": 0.90,        # tier 2: lead / podium
            "podium_battle": 0.85,
            "incident": 0.80,           # tier 3: major incidents
            "pit_strategy": 0.65,       # tier 4: pit strategy top positions
            "overtake": 0.55,           # tier 5: midfield action
            "midfield_battle": 0.50,
            "strategic": 0.40,          # tier 5b: strategic developments
            "telemetry_change": 0.15,   # tier 6: low-impact
        }
    )
