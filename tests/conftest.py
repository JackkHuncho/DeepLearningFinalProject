"""Shared test fixtures for the f1_commentary test suite."""

from pathlib import Path

import pytest

from f1_commentary.config import AppConfig


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory tree for tests."""
    for sub in ("raw", "processed", "logs/traces", "embeddings", "transcripts", "datasets", "artifacts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def sample_config(tmp_data_dir: Path) -> AppConfig:
    """Return an AppConfig pointing at temporary directories."""
    return AppConfig(
        project_root=tmp_data_dir,
        data_dir=tmp_data_dir,
        log_dir=tmp_data_dir / "logs",
        trace_dir=tmp_data_dir / "logs" / "traces",
    )
