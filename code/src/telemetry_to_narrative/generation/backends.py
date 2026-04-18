"""Generator backends — thin abstraction over local inference engines.

Each backend implements ``generate(prompt, config) → str`` and reports
its ``model_variant`` string for provenance tracking.

Available backends
==================
- **MockGeneratorBackend**: deterministic fake output for testing.
- **MLXBackend**: placeholder for Apple MLX inference (Phase 8 adapter).
- **LlamaCppBackend**: placeholder for llama.cpp / llama-cpp-python inference.
"""

from __future__ import annotations

import abc
import logging
import random
from typing import Optional

from src.telemetry_to_narrative.schemas.generated_commentary import GenerationConfig

logger = logging.getLogger(__name__)


class BaseGeneratorBackend(abc.ABC):
    """Abstract base for all generator backends."""

    @abc.abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Run inference and return raw generated text."""

    @property
    @abc.abstractmethod
    def model_variant(self) -> str:
        """Human-readable model identifier for provenance."""

    @property
    def is_available(self) -> bool:
        """True if this backend is ready to serve requests."""
        return True


# ── Mock backend ─────────────────────────────────────────────────────────

# Canned responses keyed by event type for realistic mock output.
_MOCK_TEMPLATES: dict[str, list[str]] = {
    "lead_battle": [
        "The gap at the front is tightening as {d1} comes under increasing pressure from {d0}.",
        "{d0} is pushing hard but {d1} maintains the lead with a gap of just a few tenths.",
        "Intense racing at the front between {d0} and {d1}, the gap barely changing lap after lap.",
    ],
    "podium_battle": [
        "{d0} is all over the back of {d1} for a podium position.",
        "The fight for the podium intensifies as {d0} closes on {d1}.",
    ],
    "midfield_battle": [
        "Great racing in the midfield as {d0} hassles {d1} for position.",
        "{d0} is within DRS range of {d1} and looking for a way past.",
    ],
    "overtake": [
        "{d0} makes the move! A clean overtake into the braking zone.",
        "Position change — {d0} sweeps past and moves up the order.",
    ],
    "pit_strategy": [
        "{d0} pits for fresh rubber. The strategy call could be decisive.",
        "Box, box for {d0} — switching compounds as the pit window opens.",
    ],
    "race_control": [
        "Race control intervenes — the safety car is deployed.",
        "Yellow flags waving as race control responds to an incident on track.",
    ],
    "telemetry_change": [
        "Steady running across the field, drivers managing their tires.",
        "No major changes on track as the race settles into a rhythm.",
    ],
}

_GENERIC_TEMPLATES = [
    "Action continues on track with the field closely bunched.",
    "The race is at a critical juncture with plenty still to play for.",
]


class MockGeneratorBackend(BaseGeneratorBackend):
    """Deterministic mock backend for integration testing.

    Produces canned commentary based on event type and driver names
    extracted from the prompt.  Uses a seeded RNG for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        # Extract event type from the prompt header.
        event_type = self._extract_event_type(prompt)
        drivers = self._extract_drivers(prompt)

        templates = _MOCK_TEMPLATES.get(event_type, _GENERIC_TEMPLATES)
        template = self._rng.choice(templates)

        # Fill in driver placeholders.
        d0 = drivers[0] if len(drivers) > 0 else "the driver"
        d1 = drivers[1] if len(drivers) > 1 else "his rival"
        return template.format(d0=d0, d1=d1)

    @property
    def model_variant(self) -> str:
        return "mock"

    @staticmethod
    def _extract_event_type(prompt: str) -> str:
        """Parse event type from ``[Event: xxx | ...]`` header."""
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("[Event:"):
                # [Event: lead_battle | Drivers: VER, HAM]
                part = line.split("|")[0]
                return part.replace("[Event:", "").strip().rstrip("]").strip()
        return "unknown"

    @staticmethod
    def _extract_drivers(prompt: str) -> list[str]:
        """Parse driver list from ``[Event: ... | Drivers: xxx]`` header."""
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("[Event:") and "Drivers:" in line:
                part = line.split("Drivers:")[1].rstrip("]").strip()
                if part == "field":
                    return []
                return [d.strip() for d in part.split(",") if d.strip()]
        return []


# ── MLX backend placeholder ─────────────────────────────────────────────

class MLXBackend(BaseGeneratorBackend):
    """Placeholder for Apple MLX local inference.

    Requires ``mlx-lm`` to be installed.  The model path should point to
    an MLX-format model directory, optionally with an SFT adapter.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the model on first inference call."""
        try:
            from mlx_lm import load, generate as mlx_generate  # type: ignore

            if self._adapter_path:
                logger.info(
                    "Loading MLX model %s with adapter %s",
                    self._model_path, self._adapter_path,
                )
                self._model, self._tokenizer = load(
                    self._model_path, adapter_path=self._adapter_path,
                )
            else:
                logger.info("Loading MLX model %s (no adapter)", self._model_path)
                self._model, self._tokenizer = load(self._model_path)

        except ImportError:
            raise RuntimeError(
                "MLX backend requires mlx-lm. Install with: pip install mlx-lm"
            )

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if self._model is None:
            self._load()

        from mlx_lm import generate as mlx_generate  # type: ignore

        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.max_new_tokens,
            temp=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )

    @property
    def model_variant(self) -> str:
        base = self._model_path.split("/")[-1]
        if self._adapter_path:
            return f"{base}+sft-adapter"
        return base

    @property
    def is_available(self) -> bool:
        try:
            import mlx_lm  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False


# ── llama.cpp backend placeholder ────────────────────────────────────────

class LlamaCppBackend(BaseGeneratorBackend):
    """Placeholder for llama-cpp-python local inference.

    The model path should point to a GGUF model file.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
    ) -> None:
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    def _load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore

            logger.info("Loading llama.cpp model from %s", self._model_path)
            kwargs: dict = dict(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            if self._adapter_path:
                logger.info("Applying LoRA adapter from %s", self._adapter_path)
                kwargs["lora_path"] = self._adapter_path
            self._llm = Llama(**kwargs)
        except ImportError:
            raise RuntimeError(
                "llama.cpp backend requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            )

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if self._llm is None:
            self._load()

        result = self._llm(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repetition_penalty,
        )
        return result["choices"][0]["text"]

    @property
    def model_variant(self) -> str:
        return f"gguf:{self._model_path.split('/')[-1]}"

    @property
    def is_available(self) -> bool:
        try:
            import llama_cpp  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False
