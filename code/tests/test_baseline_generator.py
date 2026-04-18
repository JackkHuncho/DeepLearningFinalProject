"""Comprehensive tests for Phase 13: Frontier baseline wrapper.

Coverage
========
- BaselineCommentary schema validation and roundtrip
- BaselineGenerationConfig defaults
- MockBaselineBackend deterministic behaviour
- OpenAICompatibleBaselineBackend scaffold (import + is_available)
- Baseline prompt builder (same facts/inferences, no storyline/retrieval)
- BaselineGenerator orchestration
- Batch generation
- Backend factory graceful fallback
- JSONL-compatible logging format (compatible with trace pipeline)
- Fairness comparison vs. local runtime prompt
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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
from src.telemetry_to_narrative.schemas.baseline_commentary import (
    BaselineCommentary,
    BaselineGenerationConfig,
)
from src.telemetry_to_narrative.baseline.backends import (
    BaseBaselineBackend,
    MockBaselineBackend,
    OpenAICompatibleBaselineBackend,
    load_baseline_backend,
)
from src.telemetry_to_narrative.baseline.prompt_builder import (
    build_baseline_prompt,
    build_baseline_prompt_from_state_summary,
    _BASELINE_INSTRUCTION,
)
from src.telemetry_to_narrative.baseline.baseline_generator import (
    BaselineGenerator,
    _clean_output,
)
from src.telemetry_to_narrative.generation.prompt_builder import build_prompt as build_local_prompt


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
                tire_age_laps=12, speed_kph=310.0, pace_trend=-0.05,
            ),
            "HAM": DriverRaceState(
                driver_number="44", abbreviation="HAM", position=2,
                lap_number=12, gap_ahead_sec=0.8, tire_compound="MEDIUM",
                tire_age_laps=12, speed_kph=308.0, pace_trend=0.15,
            ),
        },
        active_battles=[
            BattleState(
                driver_ahead="VER", driver_behind="HAM",
                gap_sec=0.8, pace_delta_sec=-0.1,
                battle_intensity=0.75, battle_duration_laps=5,
            ),
        ],
        derived_summary=DerivedSummary(leader="VER", total_drivers=2, active_battles=1),
    )


def _make_beat(
    frame_id: int = 5,
    event_type: EventType = EventType.LEAD_BATTLE,
    drivers: list[str] | None = None,
    selected: bool = True,
    storyline_id: str = "battle_HAM_VER",
) -> ScheduledBeat:
    ts = BASE_TS + timedelta(seconds=frame_id * 10)
    return ScheduledBeat(
        beat_id=f"beat_{frame_id}",
        timestamp=ts,
        event_type=event_type,
        involved_drivers=drivers if drivers is not None else ["VER", "HAM"],
        storyline_id=storyline_id,
        priority_score=0.75,
        score_breakdown=ScoreBreakdown(),
        selected_for_generation=selected,
        suppression_reason=SuppressionReason.NONE if selected else SuppressionReason.LOW_PRIORITY,
        source_frame_id=frame_id,
        description="Test baseline beat",
        output_length_budget=150,
    )


# =====================================================================
# Schema tests
# =====================================================================


class TestBaselineCommentarySchema:
    """Tests for BaselineCommentary pydantic schema."""

    def test_roundtrip(self):
        bc = BaselineCommentary(
            baseline_id="bl_abc123",
            timestamp=BASE_TS,
            beat_id="beat_5",
            storyline_id="battle_HAM_VER",
            event_type=EventType.LEAD_BATTLE,
            involved_drivers=["VER", "HAM"],
            model_name="gpt-4o-mini",
            provider="openai",
            prompt_text="[Event: lead_battle]\n...",
            raw_output_text="Verstappen leads Hamilton.",
            commentary_text="Verstappen leads Hamilton.",
            output_length=26,
            latency_ms=250.1,
        )
        json_str = bc.model_dump_json()
        restored = BaselineCommentary.model_validate_json(json_str)
        assert restored.baseline_id == "bl_abc123"
        assert restored.provider == "openai"
        assert restored.model_name == "gpt-4o-mini"
        assert restored.event_type == EventType.LEAD_BATTLE

    def test_defaults(self):
        bc = BaselineCommentary(
            baseline_id="bl_1",
            timestamp=BASE_TS,
            beat_id="b_1",
            event_type=EventType.OVERTAKE,
            model_name="mock",
            prompt_text="x",
            raw_output_text="y",
        )
        assert bc.provider == "mock"
        assert bc.commentary_text == ""
        assert bc.output_length == 0
        assert bc.latency_ms == 0.0
        assert bc.system == "baseline"
        assert bc.storyline_id is None

    def test_system_label_always_baseline(self):
        bc = BaselineCommentary(
            baseline_id="bl_1",
            timestamp=BASE_TS,
            beat_id="b_1",
            event_type=EventType.LEAD_BATTLE,
            model_name="m",
            prompt_text="p",
            raw_output_text="r",
        )
        assert bc.system == "baseline"

    def test_generation_config_default(self):
        config = BaselineGenerationConfig()
        assert config.max_new_tokens == 128
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.provider == "mock"

    def test_generation_config_override(self):
        config = BaselineGenerationConfig(
            max_new_tokens=64, temperature=0.3, provider="openai", model_name="gpt-4o-mini",
        )
        assert config.max_new_tokens == 64
        assert config.temperature == 0.3
        assert config.provider == "openai"


# =====================================================================
# Backend abstraction tests
# =====================================================================


class TestMockBaselineBackend:
    """Tests for MockBaselineBackend."""

    def test_is_base_subclass(self):
        assert issubclass(MockBaselineBackend, BaseBaselineBackend)

    def test_provider_is_mock(self):
        backend = MockBaselineBackend()
        assert backend.provider == "mock"

    def test_default_model_name(self):
        backend = MockBaselineBackend()
        assert backend.model_name == "mock-baseline"

    def test_custom_model_name(self):
        backend = MockBaselineBackend(model_name="gpt-fake")
        assert backend.model_name == "gpt-fake"

    def test_is_available(self):
        assert MockBaselineBackend().is_available is True

    def test_deterministic_with_seed(self):
        prompt = "[Event: lead_battle | Drivers: VER, HAM]\n## Observed Facts\n## Task\n..."
        config = BaselineGenerationConfig()
        out_1 = MockBaselineBackend(seed=7).generate(prompt, config)
        out_2 = MockBaselineBackend(seed=7).generate(prompt, config)
        assert out_1 == out_2

    def test_different_seeds_differ(self):
        prompt = "[Event: lead_battle | Drivers: VER, HAM]\n"
        config = BaselineGenerationConfig()
        outs = {
            MockBaselineBackend(seed=s).generate(prompt, config)
            for s in (1, 2, 3, 4, 5)
        }
        # At least some variation across seeds.
        assert len(outs) >= 2

    def test_substitutes_drivers(self):
        prompt = "[Event: overtake | Drivers: LEC, SAI]\n"
        out = MockBaselineBackend(seed=1).generate(prompt, BaselineGenerationConfig())
        assert "LEC" in out or "SAI" in out

    def test_race_control_no_drivers(self):
        prompt = "[Event: race_control | Drivers: field]\n"
        out = MockBaselineBackend(seed=1).generate(prompt, BaselineGenerationConfig())
        # Should not crash and should produce text.
        assert len(out) > 0

    def test_unknown_event_type_uses_generic(self):
        prompt = "[Event: unknown_event | Drivers: VER, HAM]\n"
        out = MockBaselineBackend(seed=1).generate(prompt, BaselineGenerationConfig())
        assert len(out) > 0


class TestOpenAIBackendScaffold:
    """Tests for the OpenAI-compatible backend scaffold (no network)."""

    def test_is_base_subclass(self):
        assert issubclass(OpenAICompatibleBaselineBackend, BaseBaselineBackend)

    def test_model_name_stored(self):
        backend = OpenAICompatibleBaselineBackend(
            model_name="gpt-4o-mini", api_key="dummy",
        )
        assert backend.model_name == "gpt-4o-mini"

    def test_default_provider_label(self):
        backend = OpenAICompatibleBaselineBackend(
            model_name="gpt-4o-mini", api_key="dummy",
        )
        assert backend.provider == "openai"

    def test_custom_provider_label(self):
        backend = OpenAICompatibleBaselineBackend(
            model_name="deepseek-chat", api_key="dummy", provider_label="deepseek",
        )
        assert backend.provider == "deepseek"

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAICompatibleBaselineBackend(model_name="gpt-4o-mini")
        # is_available depends on openai package and key presence.
        # If openai package is installed but no key, False.
        try:
            import openai  # type: ignore  # noqa: F401
            assert backend.is_available is False
        except ImportError:
            assert backend.is_available is False

    def test_generate_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        backend = OpenAICompatibleBaselineBackend(model_name="gpt-4o-mini")
        # Should raise on first use when no key available.
        try:
            import openai  # type: ignore  # noqa: F401
            with pytest.raises(RuntimeError, match="API key"):
                backend.generate("prompt", BaselineGenerationConfig())
        except ImportError:
            with pytest.raises(RuntimeError):
                backend.generate("prompt", BaselineGenerationConfig())


class TestBackendFactory:
    """Tests for load_baseline_backend."""

    def test_default_mock(self):
        backend = load_baseline_backend()
        assert isinstance(backend, MockBaselineBackend)
        assert backend.provider == "mock"

    def test_explicit_mock(self):
        backend = load_baseline_backend("mock")
        assert isinstance(backend, MockBaselineBackend)

    def test_unknown_falls_back_to_mock(self):
        backend = load_baseline_backend("totally_unknown_provider")
        assert isinstance(backend, MockBaselineBackend)

    def test_openai_with_key(self):
        backend = load_baseline_backend(
            "openai", model_name="gpt-4o-mini", api_key="dummy-key",
        )
        # Returned backend may be either OpenAI scaffold (if client loads)
        # or mock fallback if openai package unavailable — both are acceptable.
        assert isinstance(backend, BaseBaselineBackend)
        assert backend.model_name in ("gpt-4o-mini", "mock-baseline")

    def test_mock_seed_passthrough(self):
        b1 = load_baseline_backend("mock", mock_seed=99)
        b2 = load_baseline_backend("mock", mock_seed=99)
        prompt = "[Event: lead_battle | Drivers: VER, HAM]\n"
        cfg = BaselineGenerationConfig()
        assert b1.generate(prompt, cfg) == b2.generate(prompt, cfg)


# =====================================================================
# Prompt builder tests
# =====================================================================


class TestBaselinePromptBuilder:
    """Tests for build_baseline_prompt."""

    def test_contains_event_header(self):
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "[Event: lead_battle | Drivers: VER, HAM]" in prompt

    def test_contains_facts_section(self):
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "## Observed Facts" in prompt

    def test_contains_inferences_section(self):
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "## Derived Inferences" in prompt

    def test_contains_task_instruction(self):
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "## Task" in prompt
        assert _BASELINE_INSTRUCTION in prompt

    def test_excludes_storyline_section(self):
        """Fairness: baseline does NOT receive storyline context."""
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "## Storyline" not in prompt
        assert "battle_HAM_VER" not in prompt

    def test_excludes_retrieval_context(self):
        """Fairness: baseline has no retrieval context."""
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        # Context section is present but empty (from format_input_text).
        # The baseline never gets actual retrieved items.
        assert "(none available)" in prompt or "## Context" in prompt

    def test_includes_driver_facts(self):
        prompt = build_baseline_prompt(_make_snapshot(), _make_beat())
        assert "VER" in prompt
        assert "HAM" in prompt

    def test_race_control_no_drivers(self):
        prompt = build_baseline_prompt(
            _make_snapshot(),
            _make_beat(event_type=EventType.RACE_CONTROL, drivers=[]),
        )
        assert "field" in prompt

    def test_from_state_summary(self):
        prompt = build_baseline_prompt_from_state_summary(
            {"involved_drivers": {"VER": {"position": 1}}},
            EventType.LEAD_BATTLE,
            ["VER", "HAM"],
        )
        assert "## Task" in prompt
        assert "## Storyline" not in prompt


class TestPromptFairness:
    """Fairness comparison tests: baseline vs local prompt contents."""

    def test_same_facts_content(self):
        """Both prompts should include the same observed-fact section content."""
        snap = _make_snapshot()
        beat = _make_beat()
        baseline_prompt = build_baseline_prompt(snap, beat)
        local_prompt, _ = build_local_prompt(snap, beat)

        # Both should contain the key facts (position, tire).
        for fragment in ("VER", "HAM", "MEDIUM", "lap 12"):
            assert fragment in baseline_prompt, f"baseline missing {fragment}"
            assert fragment in local_prompt, f"local missing {fragment}"

    def test_baseline_lacks_storyline_local_has_it(self):
        snap = _make_snapshot()
        beat = _make_beat()
        baseline_prompt = build_baseline_prompt(snap, beat)
        local_prompt, _ = build_local_prompt(snap, beat)

        assert "## Storyline" not in baseline_prompt
        assert "## Storyline" in local_prompt
        assert beat.storyline_id in local_prompt

    def test_both_have_task_section(self):
        snap = _make_snapshot()
        beat = _make_beat()
        baseline_prompt = build_baseline_prompt(snap, beat)
        local_prompt, _ = build_local_prompt(snap, beat)
        assert "## Task" in baseline_prompt
        assert "## Task" in local_prompt


# =====================================================================
# Output cleanup tests
# =====================================================================


class TestCleanOutput:
    """Tests for the minimal cleanup function."""

    def test_strips_whitespace(self):
        assert _clean_output("  hello  ") == "hello"

    def test_strips_double_quotes(self):
        assert _clean_output('"hello"') == "hello"

    def test_strips_single_quotes(self):
        assert _clean_output("'hello'") == "hello"

    def test_passthrough(self):
        assert _clean_output("Verstappen leads.") == "Verstappen leads."

    def test_keeps_internal_quotes(self):
        assert _clean_output('He said "go" now.') == 'He said "go" now.'


# =====================================================================
# BaselineGenerator orchestration tests
# =====================================================================


class TestBaselineGenerator:
    """Tests for BaselineGenerator."""

    def test_generate_returns_baseline_commentary(self):
        backend = MockBaselineBackend(seed=1)
        gen = BaselineGenerator(backend)
        result = gen.generate(_make_snapshot(), _make_beat())
        assert isinstance(result, BaselineCommentary)
        assert result.beat_id == "beat_5"
        assert result.storyline_id == "battle_HAM_VER"
        assert result.event_type == EventType.LEAD_BATTLE
        assert result.model_name == "mock-baseline"
        assert result.provider == "mock"
        assert result.system == "baseline"

    def test_generate_populates_prompt_and_output(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        result = gen.generate(_make_snapshot(), _make_beat())
        assert "[Event: lead_battle" in result.prompt_text
        assert len(result.raw_output_text) > 0
        assert len(result.commentary_text) > 0
        assert result.output_length == len(result.raw_output_text)

    def test_latency_measured(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        result = gen.generate(_make_snapshot(), _make_beat())
        assert result.latency_ms >= 0.0

    def test_baseline_id_generated(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        r1 = gen.generate(_make_snapshot(), _make_beat())
        r2 = gen.generate(_make_snapshot(), _make_beat())
        assert r1.baseline_id != r2.baseline_id
        assert r1.baseline_id.startswith("bl_")

    def test_deterministic_output_same_seed(self):
        g1 = BaselineGenerator(MockBaselineBackend(seed=42))
        g2 = BaselineGenerator(MockBaselineBackend(seed=42))
        r1 = g1.generate(_make_snapshot(), _make_beat())
        r2 = g2.generate(_make_snapshot(), _make_beat())
        assert r1.commentary_text == r2.commentary_text

    def test_custom_config(self):
        cfg = BaselineGenerationConfig(max_new_tokens=64, temperature=0.3)
        gen = BaselineGenerator(MockBaselineBackend(), cfg)
        result = gen.generate(_make_snapshot(), _make_beat())
        assert result.generation_config.max_new_tokens == 64
        assert result.generation_config.temperature == 0.3

    def test_config_inherits_backend_identity(self):
        gen = BaselineGenerator(MockBaselineBackend(model_name="custom-mock"))
        result = gen.generate(_make_snapshot(), _make_beat())
        assert result.model_name == "custom-mock"
        assert result.generation_config.model_name == "custom-mock"

    def test_batch_filters_unselected(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        beats = [
            _make_beat(frame_id=1, selected=True),
            _make_beat(frame_id=2, selected=False),
            _make_beat(frame_id=3, selected=True),
        ]
        snaps = [_make_snapshot(frame_id=i) for i in (1, 2, 3)]
        results = gen.generate_batch(snaps, beats)
        assert len(results) == 2  # only selected

    def test_batch_max_items(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        beats = [_make_beat(frame_id=i) for i in range(1, 11)]
        snaps = [_make_snapshot(frame_id=i) for i in range(1, 11)]
        results = gen.generate_batch(snaps, beats, max_items=3)
        assert len(results) == 3

    def test_batch_skips_missing_snapshot(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        beats = [_make_beat(frame_id=1), _make_beat(frame_id=99)]
        snaps = [_make_snapshot(frame_id=1)]
        results = gen.generate_batch(snaps, beats)
        assert len(results) == 1
        assert results[0].beat_id == "beat_1"


# =====================================================================
# Trace log compatibility tests
# =====================================================================


class TestTraceLogCompatibility:
    """Ensure baseline output can flow through the trace/evaluation pipeline."""

    def test_jsonl_serializable(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        result = gen.generate(_make_snapshot(), _make_beat())
        line = result.model_dump_json()
        # Must be a single JSON object (no embedded newlines that break JSONL).
        assert "\n" not in line
        parsed = json.loads(line)
        assert parsed["system"] == "baseline"
        assert parsed["beat_id"] == "beat_5"
        assert parsed["event_type"] == "lead_battle"

    def test_jsonl_roundtrip_list(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        beats = [_make_beat(frame_id=i) for i in (1, 3, 5)]
        snaps = [_make_snapshot(frame_id=i) for i in (1, 3, 5)]
        results = gen.generate_batch(snaps, beats)

        # Write as JSONL.
        lines = [r.model_dump_json() for r in results]
        jsonl = "\n".join(lines)

        # Parse back.
        restored = [
            BaselineCommentary.model_validate_json(line)
            for line in jsonl.split("\n") if line.strip()
        ]
        assert len(restored) == len(results)
        for orig, r in zip(results, restored):
            assert orig.baseline_id == r.baseline_id
            assert orig.commentary_text == r.commentary_text
            assert orig.beat_id == r.beat_id

    def test_system_label_routable(self):
        """Downstream eval can filter by system=baseline vs system=local."""
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        result = gen.generate(_make_snapshot(), _make_beat())
        parsed = json.loads(result.model_dump_json())
        assert parsed["system"] == "baseline"

    def test_has_provider_and_model(self):
        """Trace log needs provider/model fields for A/B attribution."""
        gen = BaselineGenerator(MockBaselineBackend(model_name="gpt-test"))
        result = gen.generate(_make_snapshot(), _make_beat())
        parsed = json.loads(result.model_dump_json())
        assert parsed["provider"] == "mock"
        assert parsed["model_name"] == "gpt-test"


# =====================================================================
# Race-control event tests
# =====================================================================


class TestRaceControlEvent:
    """Baseline must handle events with no involved drivers."""

    def test_race_control_generates(self):
        gen = BaselineGenerator(MockBaselineBackend(seed=1))
        beat = _make_beat(
            event_type=EventType.RACE_CONTROL, drivers=[], storyline_id="race_control",
        )
        result = gen.generate(_make_snapshot(), beat)
        assert result.involved_drivers == []
        assert len(result.commentary_text) > 0
        assert result.storyline_id == "race_control"
