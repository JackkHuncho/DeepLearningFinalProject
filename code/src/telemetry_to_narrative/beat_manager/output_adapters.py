"""Output adapters for final commentary — terminal and SSE streaming.

Two output modes:
1. TerminalAdapter  — pretty-prints FinalCommentary to the console.
2. SSEAdapter       — emits Server-Sent Events payloads for lightweight
                      real-time streaming to a browser or dashboard.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, TextIO
import sys

from src.telemetry_to_narrative.schemas.final_commentary import (
    FinalCommentary,
    ConfidenceBand,
    ProgressionStage,
)

logger = logging.getLogger(__name__)

# ── Terminal colours (ANSI) ─────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"

_CONFIDENCE_COLOURS: dict[ConfidenceBand, str] = {
    ConfidenceBand.HIGH: _GREEN,
    ConfidenceBand.MEDIUM: _YELLOW,
    ConfidenceBand.LOW: _RED,
    ConfidenceBand.SUPPRESSED: _DIM,
}

_STAGE_SYMBOLS: dict[ProgressionStage, str] = {
    ProgressionStage.INITIAL_INTEREST: ".",
    ProgressionStage.ACTIVE_PRESSURE: "..",
    ProgressionStage.IMMINENT_ACTION: "..!",
    ProgressionStage.DECISIVE_EVENT: "!!!",
    ProgressionStage.CONCLUDED: "---",
}


class TerminalAdapter:
    """Pretty-prints FinalCommentary objects to a terminal stream.

    Usage::

        adapter = TerminalAdapter()
        for final in finals:
            adapter.emit(final)
        adapter.print_summary()
    """

    def __init__(
        self,
        stream: TextIO = sys.stdout,
        colour: bool = True,
        show_suppressed: bool = False,
    ) -> None:
        self._stream = stream
        self._colour = colour
        self._show_suppressed = show_suppressed
        self._emitted = 0
        self._suppressed = 0

    def emit(self, final: FinalCommentary) -> None:
        """Print one FinalCommentary to the terminal."""
        if final.suppression_applied and not self._show_suppressed:
            self._suppressed += 1
            return

        if final.suppression_applied:
            self._suppressed += 1
            self._print_suppressed(final)
            return

        self._emitted += 1
        self._print_active(final)

    def emit_batch(self, finals: list[FinalCommentary]) -> None:
        """Print a batch of FinalCommentary objects."""
        for f in finals:
            self.emit(f)

    def print_summary(self) -> None:
        """Print a summary line after all commentary has been emitted."""
        line = (
            f"\n--- Beat Manager Output: "
            f"{self._emitted} emitted, {self._suppressed} suppressed ---\n"
        )
        if self._colour:
            self._stream.write(f"{_BOLD}{line}{_RESET}")
        else:
            self._stream.write(line)
        self._stream.flush()

    # ── Internal ────────────────────────────────────────────────────────

    def _print_active(self, final: FinalCommentary) -> None:
        conf_col = _CONFIDENCE_COLOURS.get(final.confidence_band, "")
        stage_sym = _STAGE_SYMBOLS.get(final.progression_stage, "?")
        drivers = ", ".join(final.involved_drivers) if final.involved_drivers else "field"

        lap_str = ""
        if final.race_state_summary and final.race_state_summary.lap is not None:
            lap_str = f"L{final.race_state_summary.lap} "

        if self._colour:
            header = (
                f"{_BOLD}{_CYAN}[{final.event_type.value}]{_RESET} "
                f"{lap_str}"
                f"{_MAGENTA}{drivers}{_RESET}  "
                f"{conf_col}[{final.confidence_band.value}]{_RESET}  "
                f"{stage_sym}"
            )
            text = f"  {final.final_text}"
            meta = (
                f"{_DIM}  story={final.storyline_id}  "
                f"lag={final.replay_lag_seconds:.1f}s{_RESET}"
            )
        else:
            header = (
                f"[{final.event_type.value}] "
                f"{lap_str}"
                f"{drivers}  "
                f"[{final.confidence_band.value}]  "
                f"{stage_sym}"
            )
            text = f"  {final.final_text}"
            meta = (
                f"  story={final.storyline_id}  "
                f"lag={final.replay_lag_seconds:.1f}s"
            )

        self._stream.write(header + "\n")
        self._stream.write(text + "\n")
        self._stream.write(meta + "\n\n")
        self._stream.flush()

    def _print_suppressed(self, final: FinalCommentary) -> None:
        reason = final.suppression_reason.value
        if self._colour:
            line = (
                f"{_DIM}  [SUPPRESSED: {reason}] "
                f"{final.event_type.value} — "
                f"{final.final_text[:50]}...{_RESET}\n"
            )
        else:
            line = (
                f"  [SUPPRESSED: {reason}] "
                f"{final.event_type.value} — "
                f"{final.final_text[:50]}...\n"
            )
        self._stream.write(line)
        self._stream.flush()


class SSEAdapter:
    """Formats FinalCommentary as Server-Sent Events payloads.

    Produces text in the standard SSE wire format::

        event: commentary
        data: {"final_id": "...", ...}

    Usage::

        adapter = SSEAdapter()
        for final in finals:
            payload = adapter.format(final)
            # send payload over HTTP response or write to stream
    """

    EVENT_NAME = "commentary"
    SUPPRESSED_EVENT_NAME = "suppressed"

    def __init__(self, include_suppressed: bool = False) -> None:
        self._include_suppressed = include_suppressed

    def format(self, final: FinalCommentary) -> Optional[str]:
        """Format a FinalCommentary as an SSE payload string.

        Returns None for suppressed commentaries when include_suppressed
        is False.
        """
        if final.suppression_applied and not self._include_suppressed:
            return None

        event_name = (
            self.SUPPRESSED_EVENT_NAME
            if final.suppression_applied
            else self.EVENT_NAME
        )

        data = self._build_payload(final)
        data_json = json.dumps(data, default=str)

        return f"event: {event_name}\ndata: {data_json}\n\n"

    def format_batch(self, finals: list[FinalCommentary]) -> list[str]:
        """Format a batch, returning only non-None payloads."""
        results = []
        for f in finals:
            payload = self.format(f)
            if payload is not None:
                results.append(payload)
        return results

    def emit(self, final: FinalCommentary, stream: TextIO = sys.stdout) -> bool:
        """Format and write directly to a stream. Returns True if emitted."""
        payload = self.format(final)
        if payload is None:
            return False
        stream.write(payload)
        stream.flush()
        return True

    @staticmethod
    def _build_payload(final: FinalCommentary) -> dict:
        """Build the JSON payload for one SSE event."""
        return {
            "final_id": final.final_id,
            "timestamp": final.timestamp.isoformat(),
            "beat_id": final.beat_id,
            "storyline_id": final.storyline_id,
            "event_type": final.event_type.value,
            "involved_drivers": final.involved_drivers,
            "final_text": final.final_text,
            "confidence_band": final.confidence_band.value,
            "replay_lag_seconds": final.replay_lag_seconds,
            "progression_stage": final.progression_stage.value,
            "suppression_applied": final.suppression_applied,
            "suppression_reason": final.suppression_reason.value,
            "race_state": {
                "lap": final.race_state_summary.lap,
                "leader": final.race_state_summary.leader,
                "flag_state": final.race_state_summary.flag_state,
                "safety_car": final.race_state_summary.safety_car,
            },
        }
