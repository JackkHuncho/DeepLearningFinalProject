"""Pipeline configuration — stop points, feature flags, and mode selection.

Kept deliberately small: these are the knobs the runner exposes, not
a place to redesign any component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineMode(str, Enum):
    """Which generation path to run through."""

    LOCAL = "local"          # full local structured system
    BASELINE = "baseline"    # frontier LLM baseline (no BeatManager)


class StopAfter(str, Enum):
    """Debug stop points. Earlier stages feed later ones."""

    NONE = "none"
    REPLAY = "replay"                # stop after replay stage
    SNAPSHOTS = "snapshots"          # stop after state engine
    BEATS = "beats"                  # stop after scheduler
    GENERATION = "generation"        # stop after commentary generator
    FINAL = "final"                  # stop after beat manager (default for local)


class Scenario(str, Enum):
    """Optional event-type filter for demo modes."""

    ALL = "all"
    LEAD_BATTLE = "lead_battle"
    PIT_STRATEGY = "pit_strategy"
    RACE_CONTROL = "race_control"


@dataclass
class PipelineConfig:
    """All runner-level options in one place.

    None of these reconfigure module internals — they only gate which
    stages run and where intermediates are written.
    """

    # Mode.
    mode: PipelineMode = PipelineMode.LOCAL
    stop_after: StopAfter = StopAfter.NONE
    scenario: Scenario = Scenario.ALL

    # Feature flags.
    enable_retrieval: bool = False
    enable_grounding_guard: bool = False
    enable_beat_manager: bool = True
    enable_terminal_output: bool = True
    enable_sse_output: bool = False

    # Replay controls.
    max_frames: Optional[int] = None
    replay_speed: float = 1.0
    realtime: bool = False   # runner defaults to False (batch); CLI can override

    # Throughput.
    max_generation_items: Optional[int] = None

    # Trace / artifact paths.
    trace_dir: Optional[str] = None

    # Backend plumbing.
    local_backend: str = "mock"
    local_model_path: Optional[str] = None
    local_adapter_path: Optional[str] = None
    baseline_backend: str = "mock"
    baseline_model_name: Optional[str] = None
    baseline_api_key: Optional[str] = None
    baseline_base_url: Optional[str] = None

    # Beat manager knobs.
    beat_refresh_interval: int = 3
    include_suppressed_in_output: bool = False

    # Generation knobs.
    max_new_tokens: int = 128
    temperature: float = 0.7

    # Misc.
    session_label: Optional[str] = None  # e.g. "2024-Monza-R"

    # ── Validation ─────────────────────────────────────────────────

    # Backend names are intentionally NOT validated here — the loaders
    # fall back to ``mock`` on unknown types so a typo never aborts a run.

    def validate(self) -> None:
        """Fail fast on out-of-range or contradictory numeric values.

        Called automatically by ``PipelineRunner.run`` so CLI users get
        an early, readable error instead of a confusing traceback later.
        """
        errors: list[str] = []

        if self.max_frames is not None and self.max_frames < 0:
            errors.append(f"max_frames must be >= 0 (got {self.max_frames})")
        if self.max_generation_items is not None and self.max_generation_items < 0:
            errors.append(
                f"max_generation_items must be >= 0 (got {self.max_generation_items})"
            )
        if self.replay_speed <= 0:
            errors.append(f"replay_speed must be > 0 (got {self.replay_speed})")
        if self.beat_refresh_interval < 0:
            errors.append(
                f"beat_refresh_interval must be >= 0 (got {self.beat_refresh_interval})"
            )
        if self.max_new_tokens <= 0:
            errors.append(f"max_new_tokens must be > 0 (got {self.max_new_tokens})")
        if not 0.0 <= self.temperature <= 2.0:
            errors.append(
                f"temperature must be in [0.0, 2.0] (got {self.temperature})"
            )

        if errors:
            raise ValueError(
                "Invalid PipelineConfig:\n  - " + "\n  - ".join(errors)
            )
