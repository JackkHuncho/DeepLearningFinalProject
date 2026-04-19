"""ReplayAdapter – converts a FastF1 session into timestamped TelemetryFrames."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import fastf1

from src.telemetry_to_narrative.schemas.telemetry_frame import (
    DriverState,
    GlobalState,
    SessionInfo,
    TelemetryFrame,
)

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """Convert to float, returning None for NaN / missing."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 3)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


def _safe_bool(val) -> Optional[bool]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return bool(int(f))
    except (TypeError, ValueError):
        return None


def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(abs(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ReplayAdapter:
    """Converts historical FastF1 session data into a stream of TelemetryFrames.

    Supports two output modes:
      - **generator**: yields frames one at a time (for pipeline consumption)
      - **jsonl**: writes frames to a JSONL file on disk

    Supports replay speeds: 1x (real-time pacing) and accelerated (e.g. 5x, 10x).
    """

    def __init__(
        self,
        session: fastf1.core.Session,
        speed: float = 1.0,
        frame_interval_sec: float = 1.0,
    ):
        """
        Parameters
        ----------
        session : fastf1.core.Session
            A loaded FastF1 session.
        speed : float
            Replay speed multiplier (1.0 = real-time, 10.0 = 10x).
        frame_interval_sec : float
            Wall-clock seconds between frames at 1x speed.
        """
        self.session = session
        self.speed = max(speed, 0.01)
        self.frame_interval_sec = frame_interval_sec

        self.session_info = SessionInfo(
            year=int(session.event["EventDate"].year),
            grand_prix=str(session.event["EventName"]),
            session_type=session.name,
        )

        self._frames: list[TelemetryFrame] = []
        self._build_frames()

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _build_frames(self) -> None:
        """Pre-compute all frames from the session data."""
        laps = self.session.laps
        if laps is None or len(laps) == 0:
            logger.warning("No lap data available for this session.")
            return

        # Build a time-ordered sequence of sample points from lap boundaries.
        # Each lap has a LapStartDate; we sample the state at each unique time.
        lap_times: list[pd.Timestamp] = []
        for _, lap in laps.iterrows():
            start = lap.get("LapStartDate")
            if pd.notna(start):
                lap_times.append(pd.Timestamp(start))

        if not lap_times:
            logger.warning("Could not extract any lap start timestamps.")
            return

        lap_times = sorted(set(lap_times))

        # Weather data (merge nearest)
        weather = self.session.weather_data
        has_weather = weather is not None and len(weather) > 0

        # Race control messages for track status
        messages = self.session.race_control_messages
        has_messages = messages is not None and len(messages) > 0

        session_start = lap_times[0]

        for idx, ts in enumerate(lap_times):
            session_time = ts - session_start
            if isinstance(session_time, pd.Timedelta):
                session_time = session_time.to_pytimedelta()

            drivers = self._driver_states_at(ts, laps)
            global_state = self._global_state_at(
                ts,
                session_start,
                weather if has_weather else None,
                messages if has_messages else None,
            )

            utc_ts = ts.to_pydatetime()
            if utc_ts.tzinfo is None:
                utc_ts = utc_ts.replace(tzinfo=timezone.utc)

            frame = TelemetryFrame(
                frame_id=idx,
                timestamp=utc_ts,
                session_time=session_time,
                session_info=self.session_info,
                drivers=drivers,
                global_state=global_state,
            )
            self._frames.append(frame)

        logger.info("Built %d telemetry frames.", len(self._frames))

    def _driver_states_at(
        self, ts: pd.Timestamp, laps: pd.DataFrame
    ) -> list[DriverState]:
        """Snapshot per-driver state at a given timestamp."""
        # Find the most recent lap for each driver at or before `ts`
        candidates = laps[laps["LapStartDate"] <= ts].copy()
        if candidates.empty:
            return []

        # Keep last lap per driver
        latest = candidates.sort_values("LapStartDate").groupby("DriverNumber").last()

        # Compute positions by cumulative time if available
        if "Position" in latest.columns:
            latest = latest.sort_values("Position")

        drivers: list[DriverState] = []
        prev_time = None
        for driver_num, row in latest.iterrows():
            lap_time_val = row.get("LapTime")
            cum_time = row.get("Time")  # cumulative session time

            gap: Optional[float] = None
            if prev_time is not None and pd.notna(cum_time) and pd.notna(prev_time):
                delta = cum_time - prev_time
                if isinstance(delta, pd.Timedelta):
                    gap = round(delta.total_seconds(), 3)

            compound = row.get("Compound")
            tire_life = row.get("TyreLife")
            drs = None  # DRS not directly on lap rows; available in car telemetry

            ds = DriverState(
                driver_number=str(driver_num),
                abbreviation=str(row.get("Driver", driver_num)),
                position=_safe_int(row.get("Position")),
                lap_number=_safe_int(row.get("LapNumber")),
                gap_ahead_sec=gap if gap and gap > 0 else None,
                tire_compound=str(compound) if pd.notna(compound) else None,
                tire_age_laps=_safe_int(tire_life),
                drs_active=drs,
                speed_kph=_safe_float(row.get("SpeedST")),
            )
            drivers.append(ds)
            prev_time = cum_time

        return drivers

    def _global_state_at(
        self,
        ts: pd.Timestamp,
        session_start: pd.Timestamp,
        weather: Optional[pd.DataFrame],
        messages: Optional[pd.DataFrame],
    ) -> GlobalState:
        """Build global state snapshot for a given timestamp."""
        gs = GlobalState()

        # session-relative elapsed time as a Timedelta
        elapsed = ts - session_start

        if weather is not None and len(weather) > 0:
            if "Time" in weather.columns:
                earlier = weather[weather["Time"] <= elapsed]
                if earlier.empty:
                    earlier = weather.head(1)
                wx = earlier.iloc[-1] if not earlier.empty else weather.iloc[0]
            else:
                wx = weather.iloc[0]

            gs.air_temp_celsius = _safe_float(wx.get("AirTemp"))
            gs.track_temp_celsius = _safe_float(wx.get("TrackTemp"))
            gs.humidity_pct = _safe_float(wx.get("Humidity"))
            gs.rainfall = _safe_bool(wx.get("Rainfall"))
            gs.wind_speed_ms = _safe_float(wx.get("WindSpeed"))

        if messages is not None and len(messages) > 0:
            if "Time" in messages.columns:
                earlier_msgs = messages[messages["Time"] <= elapsed]
                if not earlier_msgs.empty:
                    last_msg = earlier_msgs.iloc[-1]
                    category = last_msg.get("Category", "")
                    if category in ("Flag", "SafetyCar", "TrackStatus"):
                        msg_text = str(last_msg.get("Message", ""))
                        gs.track_status = msg_text
                        gs.safety_car = "safety car" in msg_text.lower()

        return gs

    # ------------------------------------------------------------------
    # Output modes
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def replay(
        self,
        max_frames: Optional[int] = None,
        realtime: bool = True,
    ) -> Generator[TelemetryFrame, None, None]:
        """Yield frames one at a time, optionally paced by replay speed.

        Parameters
        ----------
        max_frames : int | None
            Stop after this many frames (``None`` = all).
        realtime : bool
            If True, sleep between frames to simulate replay speed.
            If False, yield as fast as possible (useful for batch processing).
        """
        limit = max_frames if max_frames else len(self._frames)
        sleep_sec = self.frame_interval_sec / self.speed

        for i, frame in enumerate(self._frames[:limit]):
            yield frame
            if realtime and i < limit - 1:
                time.sleep(sleep_sec)

    def to_jsonl(
        self,
        output_path: Path | str,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Write frames to a JSONL file.

        Parameters
        ----------
        output_path : Path | str
            Destination file path.
        max_frames : int | None
            Limit number of frames written.

        Returns
        -------
        Path
            The path to the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        limit = max_frames if max_frames else len(self._frames)
        count = 0

        with open(path, "w") as fh:
            for frame in self._frames[:limit]:
                fh.write(frame.model_dump_json() + "\n")
                count += 1

        logger.info("Wrote %d frames to %s", count, path)
        return path
