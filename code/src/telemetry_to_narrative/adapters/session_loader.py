"""FastF1 session loader with local caching."""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fastf1_cache"

SESSION_TYPE_MAP = {
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
    "Q": "Q",
    "R": "R",
    "S": "S",       # Sprint
    "SQ": "SQ",     # Sprint Qualifying
}


def enable_cache(cache_dir: Path | str | None = None) -> Path:
    """Enable FastF1 disk cache and return the cache directory."""
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    logger.info("FastF1 cache enabled at %s", cache_path)
    return cache_path


def load_session(
    year: int,
    grand_prix: str,
    session_type: str,
    cache_dir: Path | str | None = None,
) -> fastf1.core.Session:
    """Download / load a FastF1 session.

    Parameters
    ----------
    year : int
        Season year (e.g. 2024).
    grand_prix : str
        Grand Prix name or round number (e.g. "Monza", "Italian Grand Prix", 14).
    session_type : str
        One of FP1, FP2, FP3, Q, R, S, SQ.
    cache_dir : Path | str | None
        Override for the cache directory. Uses default if ``None``.

    Returns
    -------
    fastf1.core.Session
        A fully loaded session with laps, telemetry, and weather.
    """
    enable_cache(cache_dir)

    mapped = SESSION_TYPE_MAP.get(session_type.upper())
    if mapped is None:
        raise ValueError(
            f"Unknown session type '{session_type}'. "
            f"Choose from: {', '.join(SESSION_TYPE_MAP)}"
        )

    logger.info("Loading session: %d %s %s", year, grand_prix, mapped)
    session = fastf1.get_session(year, grand_prix, mapped)
    session.load(
        laps=True,
        telemetry=True,
        weather=True,
        messages=True,
    )
    logger.info(
        "Session loaded: %s – %s (%d laps across %d drivers)",
        session.event["EventName"],
        session.name,
        len(session.laps) if session.laps is not None else 0,
        session.laps["DriverNumber"].nunique() if session.laps is not None and len(session.laps) > 0 else 0,
    )
    return session
