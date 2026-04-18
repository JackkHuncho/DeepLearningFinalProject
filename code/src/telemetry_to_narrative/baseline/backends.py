"""Baseline provider backends — abstraction over frontier LLM providers.

Each backend implements ``generate(prompt, config) -> str`` and reports
its ``model_name`` / ``provider`` for provenance tracking.

Available backends
==================
- **MockBaselineBackend**: deterministic fake baseline output for testing.
- **OpenAICompatibleBaselineBackend**: scaffold for any OpenAI-style
  Chat Completions API (OpenAI, Azure OpenAI, DeepSeek, Together, etc.).
  Kept as a scaffold — actual API credentials are externalized.
"""

from __future__ import annotations

import abc
import logging
import os
import random
from typing import Optional

from src.telemetry_to_narrative.schemas.baseline_commentary import (
    BaselineGenerationConfig,
)

logger = logging.getLogger(__name__)


class BaseBaselineBackend(abc.ABC):
    """Abstract base for all baseline provider backends."""

    @abc.abstractmethod
    def generate(self, prompt: str, config: BaselineGenerationConfig) -> str:
        """Run inference and return raw generated text."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for provenance."""

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Provider family: 'openai', 'anthropic', 'google', 'mock', ..."""

    @property
    def is_available(self) -> bool:
        """True if this backend is ready to serve requests."""
        return True


# ── Mock backend ─────────────────────────────────────────────────────────

# Canned baseline-style responses keyed by event type. Intentionally a bit
# more verbose / less structured than the local system's mock templates,
# to reflect what a general-purpose frontier model tends to produce when
# given the same structured prompt without system-specific constraints.
_BASELINE_MOCK_TEMPLATES: dict[str, list[str]] = {
    "lead_battle": [
        "At the front of the field, {d0} continues to pressure {d1}, with the gap hovering just under one second.",
        "A tense battle for the lead is unfolding between {d1} and {d0}, and the margin has barely moved in recent laps.",
    ],
    "podium_battle": [
        "{d0} looks poised to challenge {d1} for a podium spot, running well within DRS range on this lap.",
        "The fight for the final podium place is on, with {d0} applying sustained pressure on {d1}.",
    ],
    "midfield_battle": [
        "Through the midfield, {d0} is all over the back of {d1} and clearly looking for an opportunity to pass.",
        "In the midfield scrap, {d0} has closed right up to {d1} and the two are separated by only a handful of tenths.",
    ],
    "overtake": [
        "{d0} makes a clean move past {d1}, completing the overtake into the braking zone.",
        "A decisive manoeuvre from {d0}, sweeping past {d1} and climbing up the order.",
    ],
    "pit_strategy": [
        "{d0} has committed to a pit stop, rolling the dice on strategy as the window opens.",
        "Into the pits for {d0}, taking on fresh tyres in what could prove a pivotal call.",
    ],
    "race_control": [
        "Race control has intervened — the safety car has been deployed and the field will be neutralised.",
        "An incident on track has prompted race control to act, with yellow flags now shown in the affected sector.",
    ],
    "telemetry_change": [
        "The pace has settled for now, with drivers managing their tyres and looking after the cars.",
        "A relatively quiet phase of the race as the field steadies into a rhythm.",
    ],
}

_BASELINE_GENERIC_TEMPLATES = [
    "The race continues to develop, with plenty still at stake across the field.",
    "A complex moment in the race, with several storylines unfolding simultaneously.",
]


class MockBaselineBackend(BaseBaselineBackend):
    """Deterministic mock baseline backend for tests and offline demos.

    Produces plausible frontier-LLM-style output without requiring API
    credentials.  Uses a seeded RNG for reproducibility.
    """

    def __init__(self, seed: int = 42, model_name: str = "mock-baseline") -> None:
        self._rng = random.Random(seed)
        self._model_name = model_name

    def generate(self, prompt: str, config: BaselineGenerationConfig) -> str:
        event_type = self._extract_event_type(prompt)
        drivers = self._extract_drivers(prompt)

        templates = _BASELINE_MOCK_TEMPLATES.get(event_type, _BASELINE_GENERIC_TEMPLATES)
        template = self._rng.choice(templates)

        d0 = drivers[0] if len(drivers) > 0 else "the leading driver"
        d1 = drivers[1] if len(drivers) > 1 else "his closest rival"
        return template.format(d0=d0, d1=d1)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "mock"

    @staticmethod
    def _extract_event_type(prompt: str) -> str:
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("[Event:"):
                part = line.split("|")[0]
                return part.replace("[Event:", "").strip().rstrip("]").strip()
        return "unknown"

    @staticmethod
    def _extract_drivers(prompt: str) -> list[str]:
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("[Event:") and "Drivers:" in line:
                part = line.split("Drivers:")[1].rstrip("]").strip()
                if part == "field":
                    return []
                return [d.strip() for d in part.split(",") if d.strip()]
        return []


# ── OpenAI-compatible backend scaffold ───────────────────────────────────

class OpenAICompatibleBaselineBackend(BaseBaselineBackend):
    """Scaffold for any OpenAI-compatible Chat Completions endpoint.

    Works with OpenAI, Azure OpenAI, DeepSeek, Together, LocalAI, etc.
    Credentials and base URL are read from the environment so they stay
    out of code and version control:

    - ``OPENAI_API_KEY``  — API key (or equivalent for compatible providers)
    - ``OPENAI_BASE_URL`` — optional base URL override (defaults to OpenAI)

    The ``openai`` package is imported lazily so the module remains
    importable in environments without it installed.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_label: str = "openai",
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._system_prompt = system_prompt or (
            "You are an experienced Formula 1 commentator. "
            "Produce one short, professional commentary line grounded in "
            "the race state provided. Keep it to one or two sentences."
        )
        self._provider_label = provider_label
        self._client = None

    def _load(self) -> None:
        """Lazy-load the OpenAI client on first use."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI-compatible backend requires the 'openai' package. "
                "Install with: pip install openai"
            ) from exc

        if not self._api_key:
            raise RuntimeError(
                "OpenAI-compatible backend requires an API key. "
                "Set OPENAI_API_KEY or pass api_key= explicitly."
            )

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def generate(self, prompt: str, config: BaselineGenerationConfig) -> str:
        if self._client is None:
            self._load()

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        content = response.choices[0].message.content or ""
        return content

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_label

    @property
    def is_available(self) -> bool:
        try:
            import openai  # type: ignore  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)


# ── Backend factory ──────────────────────────────────────────────────────

def load_baseline_backend(
    backend_type: str = "mock",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_label: Optional[str] = None,
    mock_seed: int = 42,
) -> BaseBaselineBackend:
    """Load a baseline backend by name with graceful fallback to mock.

    Parameters
    ----------
    backend_type : str
        One of 'mock', 'openai'.
    model_name : str | None
        Provider-specific model name (e.g. 'gpt-4o-mini').
    api_key, base_url : str | None
        Optional credentials. Fall back to env vars.
    provider_label : str | None
        Override provider label for logging (e.g. 'azure', 'deepseek').
    mock_seed : int
        Seed for deterministic mock output.
    """
    backend_type = backend_type.lower()

    if backend_type == "mock":
        return MockBaselineBackend(
            seed=mock_seed,
            model_name=model_name or "mock-baseline",
        )

    if backend_type == "openai":
        if not model_name:
            logger.warning("No model_name given for OpenAI backend; defaulting to gpt-4o-mini.")
            model_name = "gpt-4o-mini"
        try:
            return OpenAICompatibleBaselineBackend(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                provider_label=provider_label or "openai",
            )
        except RuntimeError as exc:
            logger.warning(
                "OpenAI baseline backend unavailable (%s); falling back to mock.", exc,
            )
            return MockBaselineBackend(seed=mock_seed)

    logger.warning("Unknown baseline backend '%s'; falling back to mock.", backend_type)
    return MockBaselineBackend(seed=mock_seed)
