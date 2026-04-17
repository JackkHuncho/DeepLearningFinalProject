"""Comprehensive tests for Phase 9: Runtime commentary generator.

Coverage
========
- GeneratedCommentary schema validation
- GenerationConfig defaults and overrides
- MockGeneratorBackend behavior
- Prompt building (SFT format alignment, sections, drivers, event types)
- Missing retrieval fallback
- Missing adapter fallback (model_loader)
- Output length budgeting
- CommentaryGenerator orchestration
- Batch generation
- Backend abstraction
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telemetry_to_narrative.schemas.telemetry_frame import SessionInfo
from src.telemetry_to_narrative.schemas.candidate_event import EventType
from src.telemetry_to_narrative.schemas.race_state import (
    BattleState,
    DerivedSummary,
    DriverRaceState,
    GlobalRaceStateSummary,
    RaceStateSnapshot,
)
from src.telemetry_to_narrative.schemas.scheduled_beat import (
    ScoreBreakdown,
    ScheduledBeat,
    SuppressionReason,
)
from src.telemetry_to_narrative.schemas.generated_commentary import (
    GeneratedCommentary,
    GenerationConfig,
)
from src.telemetry_to_narrative.generation.backends import (
    BaseGeneratorBackend,
    MockGeneratorBackend,
)
from src.telemetry_to_narrative.generation.model_loader import load_backend
from src.telemetry_to_narrative.generation.prompt_builder import build_prompt
from src.telemetry_to_narrative.generation.commentary_generator import (
    CommentaryGenerator,
    _clean_output,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
SESSION_INFO = SessionInfo(year=2024, grand_prix="Monza", session_type="R")


def _make_snapshot(frame_id: int = 5, ts_offset_sec: int = 50) -> RaceStateSnapshot:
    ts = BASE_TS + timedelta(seconds=ts_offset_sec)
    return RaceStateSnapshot(
        frame_id=frame_id,
        timestamp=ts,
        session_info=SESSION_INFO,
        global_state=GlobalRaceStateSummary(track_status="GREEN FLAG"),
        drivers={
            "VER": DriverRaceState(
                driver_number="1", abbreviation="VER", position=1,
                lap_number=12, gap_ahead_sec=0.0, tire_compound="MEDIUM",
                tire_age_laps=12, speed_kph=310.0,
                pace_trend=-0.05, degradation_score=0.2, pit_window_open=False,
            ),
            "HAM": DriverRaceState(
                driver_number="44", abbreviation="HAM", position=2,
                lap_number=12, gap_ahead_sec=0.8, tire_compound="MEDIUM",
                tire_age_laps=12, speed_kph=308.0,
                pace_trend=0.15, degradation_score=0.6, pit_window_open=True,
            ),
        },
        active_battles=[
            BattleState(
                driver_ahead="VER", driver_behind="HAM",
                gap_sec=0.8, pace_delta_sec=-0.1, battle_intensity=0.75,
                battle_duration_laps=5,
            ),
        ],
        derived_summary=DerivedSummary(
            leader="VER", total_drivers=2, active_battles=1,
        ),
    )


def _make_beat(
    frame_id: int = 5,
    event_type: EventType = EventType.LEAD_BATTLE,
    drivers: list[str] | None = None,
    selected: bool = True,
    budget: int = 150,
) -> ScheduledBeat:
    ts = BASE_TS + timedelta(seconds=frame_id * 10)
    return ScheduledBeat(
        beat_id=f"beat_{frame_id}",
        timestamp=ts,
        event_type=event_type,
        involved_drivers=drivers if drivers is not None else ["VER", "HAM"],
        storyline_id="battle_HAM_VER",
        priority_score=0.75,
        score_breakdown=ScoreBreakdown(),
        selected_for_generation=selected,
        suppression_reason=SuppressionReason.NONE if selected else SuppressionReason.LOW_PRIORITY,
        source_frame_id=frame_id,
        description="VER vs HAM lead battle",
        output_length_budget=budget,
    )


# =====================================================================
# Schema tests
# =====================================================================


class TestGeneratedCommentarySchema:
    """Tests for GeneratedCommentary pydantic schema."""

    def test_roundtrip(self):
        commentary = GeneratedCommentary(
            generation_id="gen_abc123",
            timestamp=BASE_TS,
            beat_id="beat_5",
            storyline_id="battle_HAM_VER",
            event_type=EventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
            model_variant="mock",
            prompt_text="[Event: lead_battle]\n...",
            raw_output_text="Verstappen leads Hamilton.",
            commentary_text="Verstappen leads Hamilton.",
            output_length=26,
            latency_ms=1.5,
            retrieved_context_used=False,
        )
        json_str = commentary.model_dump_json()
        restored = GeneratedCommentary.model_validate_json(json_str)
        assert restored.generation_id == "gen_abc123"
        assert restored.event_type == EventType.LEAD_BATTLE
        assert restored.latency_ms == 1.5
        assert restored.commentary_text == "Verstappen leads Hamilton."

    def test_generation_config_defaults(self):
        config = GenerationConfig()
        assert config.max_new_tokens == 128
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1
        assert config.backend == "mock"

    def test_generation_config_override(self):
        config = GenerationConfig(max_new_tokens=64, temperature=0.3)
        assert config.max_new_tokens == 64
        assert config.temperature == 0.3


# =====================================================================
# Mock backend tests
# =====================================================================


class TestMockBackend:
    """Tests for MockGeneratorBackend."""

    def test_model_variant(self):
        backend = MockGeneratorBackend()
        assert backend.model_variant == "mock"

    def test_is_available(self):
        backend = MockGeneratorBackend()
        assert backend.is_available is True

    def test_deterministic_with_same_seed(self):
        b1 = MockGeneratorBackend(seed=42)
        b2 = MockGeneratorBackend(seed=42)
        prompt = "[Event: lead_battle | Drivers: VER, HAM]\n## Observed Facts\n- test"
        config = GenerationConfig()
        assert b1.generate(prompt, config) == b2.generate(prompt, config)

    def test_different_seed_may_differ(self):
        b1 = MockGeneratorBackend(seed=42)
        b2 = MockGeneratorBackend(seed=99)
        prompt = "[Event: lead_battle | Drivers: VER, HAM]\n## Observed Facts\n- test"
        config = GenerationConfig()
        # Not guaranteed to differ with only 3 templates, but output is valid either way.
        out1 = b1.generate(prompt, config)
        out2 = b2.generate(prompt, config)
        assert isinstance(out1, str) and len(out1) > 0
        assert isinstance(out2, str) and len(out2) > 0

    def test_event_type_extraction(self):
        assert MockGeneratorBackend._extract_event_type(
            "[Event: overtake | Drivers: NOR]"
        ) == "overtake"
        assert MockGeneratorBackend._extract_event_type(
            "[Event: pit_strategy | Drivers: SAI]"
        ) == "pit_strategy"
        assert MockGeneratorBackend._extract_event_type("no header") == "unknown"

    def test_driver_extraction(self):
        drivers = MockGeneratorBackend._extract_drivers(
            "[Event: lead_battle | Drivers: VER, HAM]"
        )
        assert drivers == ["VER", "HAM"]

    def test_driver_extraction_field(self):
        drivers = MockGeneratorBackend._extract_drivers(
            "[Event: race_control | Drivers: field]"
        )
        assert drivers == []

    def test_generates_relevant_content(self):
        backend = MockGeneratorBackend(seed=42)
        prompt = "[Event: pit_strategy | Drivers: SAI]\n## Observed Facts\n- SAI pitted"
        config = GenerationConfig()
        output = backend.generate(prompt, config)
        # Output should mention SAI (from template formatting).
        assert "SAI" in output

    def test_race_control_no_drivers(self):
        backend = MockGeneratorBackend(seed=42)
        prompt = "[Event: race_control | Drivers: field]\n## Observed Facts\n- SC deployed"
        config = GenerationConfig()
        output = backend.generate(prompt, config)
        assert isinstance(output, str) and len(output) > 0


# =====================================================================
# Model loader tests
# =====================================================================


class TestModelLoader:
    """Tests for load_backend with fallback behavior."""

    def test_mock_backend(self):
        backend = load_backend("mock")
        assert backend.model_variant == "mock"

    def test_unknown_backend_falls_back_to_mock(self):
        backend = load_backend("nonexistent")
        assert backend.model_variant == "mock"

    def test_mlx_without_path_falls_back(self):
        backend = load_backend("mlx", model_path=None)
        assert backend.model_variant == "mock"

    def test_llama_cpp_without_path_falls_back(self):
        backend = load_backend("llama_cpp", model_path=None)
        assert backend.model_variant == "mock"

    def test_missing_adapter_falls_back_gracefully(self):
        backend = load_backend(
            "mock",
            adapter_path="/nonexistent/adapter/path",
        )
        # Mock backend doesn't use adapters, but the loader should
        # handle the missing path warning without crashing.
        assert backend.model_variant == "mock"

    def test_mlx_without_library_falls_back(self):
        # MLX backend should fall back to mock if mlx-lm isn't installed.
        backend = load_backend("mlx", model_path="/some/model")
        # On most test environments mlx-lm isn't installed.
        assert isinstance(backend, (MockGeneratorBackend,)) or backend.is_available


# =====================================================================
# Prompt builder tests
# =====================================================================


class TestPromptBuilder:
    """Tests for inference prompt construction."""

    def test_prompt_contains_sft_format_sections(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        prompt, _ = build_prompt(snapshot, beat)

        assert "[Event: lead_battle | Drivers: VER, HAM]" in prompt
        assert "## Observed Facts" in prompt
        assert "## Derived Inferences" in prompt
        assert "## Context" in prompt
        assert "## Storyline" in prompt
        assert "## Task" in prompt

    def test_prompt_contains_driver_facts(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        prompt, _ = build_prompt(snapshot, beat)

        assert "VER" in prompt
        assert "HAM" in prompt
        assert "P1" in prompt or "P2" in prompt

    def test_prompt_contains_task_instruction(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        prompt, _ = build_prompt(snapshot, beat)

        assert "Generate one short" in prompt
        assert "commentary" in prompt.lower()

    def test_no_retrieval_context(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        prompt, retrieval_used = build_prompt(snapshot, beat)

        assert retrieval_used is False
        assert "(none available)" in prompt

    def test_with_retrieval_context(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        ctx = ["Hamilton won at Monza in 2021", "VER has a 30-point championship lead"]
        prompt, retrieval_used = build_prompt(snapshot, beat, retrieved_context=ctx)

        assert retrieval_used is True
        assert "Hamilton won at Monza in 2021" in prompt
        assert "(none available)" not in prompt

    def test_storyline_section(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        prompt, _ = build_prompt(snapshot, beat)

        assert "battle_HAM_VER" in prompt
        assert "(ongoing)" in prompt

    def test_no_storyline(self):
        snapshot = _make_snapshot()
        beat = _make_beat()
        beat.storyline_id = None
        prompt, _ = build_prompt(snapshot, beat)

        assert "(no active storyline)" in prompt

    def test_short_budget_uses_shorter_instruction(self):
        snapshot = _make_snapshot()
        beat = _make_beat(budget=50)  # short budget
        prompt, _ = build_prompt(snapshot, beat)

        # Short instruction should be used.
        assert "concise" in prompt.lower()

    def test_normal_budget_uses_full_instruction(self):
        snapshot = _make_snapshot()
        beat = _make_beat(budget=150)
        prompt, _ = build_prompt(snapshot, beat)

        assert "one or two sentences" in prompt.lower()

    def test_different_event_types(self):
        snapshot = _make_snapshot()

        for etype in [EventType.OVERTAKE, EventType.PIT_STRATEGY, EventType.RACE_CONTROL]:
            beat = _make_beat(event_type=etype, drivers=["VER"])
            prompt, _ = build_prompt(snapshot, beat)
            assert f"[Event: {etype.value}" in prompt

    def test_race_control_event_no_drivers(self):
        snapshot = _make_snapshot()
        beat = _make_beat(event_type=EventType.RACE_CONTROL, drivers=[])
        prompt, _ = build_prompt(snapshot, beat)

        assert "Drivers: field" in prompt


# =====================================================================
# Commentary generator tests
# =====================================================================


class TestCommentaryGenerator:
    """Tests for CommentaryGenerator orchestration."""

    def test_generate_single(self):
        backend = MockGeneratorBackend(seed=42)
        generator = CommentaryGenerator(backend)
        snapshot = _make_snapshot()
        beat = _make_beat()

        result = generator.generate(snapshot, beat)

        assert isinstance(result, GeneratedCommentary)
        assert result.beat_id == "beat_5"
        assert result.event_type == EventType.LEAD_BATTLE
        assert result.model_variant == "mock"
        assert result.prompt_text  # non-empty
        assert result.raw_output_text  # non-empty
        assert result.commentary_text  # non-empty
        assert result.output_length > 0
        assert result.latency_ms >= 0
        assert result.generation_config.backend == "mock"

    def test_prompt_is_always_logged(self):
        """The prompt_text field must always be populated."""
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        snapshot = _make_snapshot()
        beat = _make_beat()

        result = generator.generate(snapshot, beat)

        assert "## Observed Facts" in result.prompt_text
        assert "[Event:" in result.prompt_text

    def test_retrieval_flag_false_without_context(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        result = generator.generate(_make_snapshot(), _make_beat())
        assert result.retrieved_context_used is False

    def test_retrieval_flag_true_with_context(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        result = generator.generate(
            _make_snapshot(), _make_beat(),
            retrieved_context=["some historical fact"],
        )
        assert result.retrieved_context_used is True

    def test_output_length_budget_respected(self):
        backend = MockGeneratorBackend()
        config = GenerationConfig(max_new_tokens=200)
        generator = CommentaryGenerator(backend, config)
        beat = _make_beat(budget=80)

        result = generator.generate(_make_snapshot(), beat)

        # Config should use the beat's budget (80) not the generator's (200).
        assert result.generation_config.max_new_tokens == 80

    def test_short_budget_triggers_concise_prompt(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        beat = _make_beat(budget=50)

        result = generator.generate(_make_snapshot(), beat)

        assert "concise" in result.prompt_text.lower()

    def test_clean_output_strips_whitespace(self):
        assert _clean_output("  hello world  ") == "hello world"

    def test_clean_output_strips_quotes(self):
        assert _clean_output('"quoted text"') == "quoted text"
        assert _clean_output("'single quoted'") == "single quoted"

    def test_involved_drivers_preserved(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        beat = _make_beat(drivers=["NOR", "LEC"])
        snapshot = _make_snapshot()

        result = generator.generate(snapshot, beat)
        assert result.involved_drivers == ["NOR", "LEC"]

    def test_storyline_id_preserved(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        result = generator.generate(_make_snapshot(), _make_beat())
        assert result.storyline_id == "battle_HAM_VER"


# =====================================================================
# Batch generation tests
# =====================================================================


class TestBatchGeneration:
    """Tests for generate_batch."""

    def test_batch_processes_selected_only(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)

        snaps = [_make_snapshot(i, i * 10) for i in range(5)]
        beats = [
            _make_beat(frame_id=1, selected=True),
            _make_beat(frame_id=2, selected=False),
            _make_beat(frame_id=3, selected=True),
        ]

        results = generator.generate_batch(snaps, beats)
        assert len(results) == 2
        assert results[0].beat_id == "beat_1"
        assert results[1].beat_id == "beat_3"

    def test_batch_max_items(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)

        snaps = [_make_snapshot(i, i * 10) for i in range(10)]
        beats = [_make_beat(frame_id=i, selected=True) for i in range(10)]

        results = generator.generate_batch(snaps, beats, max_items=3)
        assert len(results) == 3

    def test_batch_skips_missing_snapshots(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)

        snaps = [_make_snapshot(0, 0)]  # only frame 0
        beats = [
            _make_beat(frame_id=0, selected=True),
            _make_beat(frame_id=99, selected=True),  # no snapshot
        ]

        results = generator.generate_batch(snaps, beats)
        assert len(results) == 1
        assert results[0].beat_id == "beat_0"

    def test_batch_empty_beats(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        results = generator.generate_batch([_make_snapshot()], [])
        assert len(results) == 0

    def test_batch_with_retrieved_contexts(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)

        snaps = [_make_snapshot(1, 10)]
        beats = [_make_beat(frame_id=1, selected=True)]

        contexts = {"beat_1": ["Hamilton's last win was 2021 British GP"]}
        results = generator.generate_batch(snaps, beats, retrieved_contexts=contexts)

        assert len(results) == 1
        assert results[0].retrieved_context_used is True
        assert "Hamilton's last win" in results[0].prompt_text


# =====================================================================
# Schema output validation
# =====================================================================


class TestOutputValidation:
    """Tests verifying GeneratedCommentary output completeness."""

    def test_all_required_fields_populated(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        result = generator.generate(_make_snapshot(), _make_beat())

        # Check all required fields are non-default / populated.
        assert result.generation_id  # non-empty string
        assert result.timestamp  # datetime populated
        assert result.beat_id  # from beat
        assert result.event_type  # from beat
        assert result.model_variant == "mock"
        assert result.prompt_text  # non-empty
        assert result.raw_output_text  # non-empty
        assert result.commentary_text  # non-empty
        assert result.output_length > 0
        assert result.latency_ms >= 0.0
        assert isinstance(result.retrieved_context_used, bool)
        assert isinstance(result.generation_config, GenerationConfig)

    def test_generation_config_in_output(self):
        backend = MockGeneratorBackend()
        config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.3,
            top_p=0.85,
        )
        generator = CommentaryGenerator(backend, config)
        beat = _make_beat(budget=64)
        result = generator.generate(_make_snapshot(), beat)

        assert result.generation_config.max_new_tokens == 64
        assert result.generation_config.temperature == 0.3
        assert result.generation_config.top_p == 0.85

    def test_output_serializes_to_jsonl(self):
        backend = MockGeneratorBackend()
        generator = CommentaryGenerator(backend)
        result = generator.generate(_make_snapshot(), _make_beat())

        json_str = result.model_dump_json()
        restored = GeneratedCommentary.model_validate_json(json_str)
        assert restored.generation_id == result.generation_id
        assert restored.commentary_text == result.commentary_text
