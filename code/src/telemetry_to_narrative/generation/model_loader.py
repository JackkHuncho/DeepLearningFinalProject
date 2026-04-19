"""Model loader — resolves backend from config with adapter fallback.

Tries to load a real backend (MLX or llama.cpp) if configured.
Falls back to MockGeneratorBackend if:
- no model path is provided
- the required library is not installed
- the adapter is missing (logs a warning, loads base model)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.telemetry_to_narrative.generation.backends import (
    BaseGeneratorBackend,
    LlamaCppBackend,
    MLXBackend,
    MockGeneratorBackend,
)

logger = logging.getLogger(__name__)


def load_backend(
    backend_type: str = "mock",
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    mock_seed: int = 42,
) -> BaseGeneratorBackend:
    """Resolve and return the appropriate generator backend.

    Parameters
    ----------
    backend_type : str
        One of ``"mock"``, ``"mlx"``, ``"llama_cpp"``.
    model_path : str | None
        Path to the base model (required for non-mock backends).
    adapter_path : str | None
        Path to an SFT adapter.  If the path does not exist, a warning
        is logged and the base model is loaded without it.
    mock_seed : int
        Seed for the mock backend's RNG.

    Returns
    -------
    BaseGeneratorBackend
        Ready-to-use backend instance.
    """
    if backend_type == "mock":
        logger.info("Using mock generator backend (seed=%d).", mock_seed)
        return MockGeneratorBackend(seed=mock_seed)

    # Validate adapter path — graceful fallback if missing.
    resolved_adapter: Optional[str] = None
    if adapter_path is not None:
        if Path(adapter_path).exists():
            resolved_adapter = adapter_path
            logger.info("SFT adapter found at %s", adapter_path)
        else:
            logger.warning(
                "SFT adapter not found at %s — loading base model only.",
                adapter_path,
            )

    if backend_type == "mlx":
        if model_path is None:
            logger.warning("No model_path for MLX backend — falling back to mock.")
            return MockGeneratorBackend(seed=mock_seed)
        backend = MLXBackend(model_path, adapter_path=resolved_adapter)
        if not backend.is_available:
            logger.warning("mlx-lm not installed — falling back to mock backend.")
            return MockGeneratorBackend(seed=mock_seed)
        return backend

    if backend_type == "llama_cpp":
        if model_path is None:
            logger.warning("No model_path for llama.cpp backend — falling back to mock.")
            return MockGeneratorBackend(seed=mock_seed)
        backend = LlamaCppBackend(model_path)
        if not backend.is_available:
            logger.warning("llama-cpp-python not installed — falling back to mock backend.")
            return MockGeneratorBackend(seed=mock_seed)
        return backend

    logger.warning("Unknown backend type '%s' — falling back to mock.", backend_type)
    return MockGeneratorBackend(seed=mock_seed)
