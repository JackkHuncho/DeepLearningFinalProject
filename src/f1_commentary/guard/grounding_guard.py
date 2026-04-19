"""Grounding guard that validates generated commentary.

Compares generated text against source telemetry and known facts
to flag or soften ungrounded claims before output.
Implemented in Phase 10.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional

from f1_commentary.schemas.events import (
    GeneratedCommentary,
    GuardedCommentary,
    ClaimSupport,
    RaceStateSnapshot,
    RetrievalResult,
    EventType,
)
from f1_commentary.config import GuardConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known F1 driver codes  (3-letter abbreviations used in timing screens)
# ---------------------------------------------------------------------------
DRIVER_CODES: set[str] = {
    "HAM", "VER", "LEC", "SAI", "NOR", "PIA", "RUS", "PER",
    "ALO", "STR", "GAS", "OCO", "TSU", "RIC", "HUL", "MAG",
    "BOT", "ZHO", "ALB", "SAR", "LAW", "BEA", "DOO", "COL",
    "ANT", "HAD", "BOR", "IAP",
}

# Tire compound names recognised by the guard
TIRE_COMPOUNDS: set[str] = {"SOFT", "MEDIUM", "HARD", "INTER", "WET"}

# Regex helpers ---------------------------------------------------------------

_POSITION_RE = re.compile(
    r"\b(?:P(\d{1,2}))\b"               # P1, P12, ...
    r"|"
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)

_ORDINAL_MAP: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

_GAP_RE = re.compile(
    r"(\d+\.?\d*)\s*(?:s(?:econds?)?)\s*(?:gap|behind|ahead)?",
    re.IGNORECASE,
)

_TIRE_RE = re.compile(
    r"\b(soft|medium|hard|inter(?:mediate)?|wet)\s*(?:tires?|tyres?|compound)?\b",
    re.IGNORECASE,
)

_TIRE_AGE_RE = re.compile(
    r"(\d+)\s*laps?\s*old",
    re.IGNORECASE,
)

_LAP_RE = re.compile(
    r"\blap\s+(\d+)\b",
    re.IGNORECASE,
)

_DRIVER_CODE_RE = re.compile(
    r"\b([A-Z]{3})\b",
)

# Predictive / assertive phrases that should be softened when unsupported
_ASSERTIVE_RE = re.compile(
    r"\b(will overtake|will win|will pit|is going to|WILL)\b",
    re.IGNORECASE,
)


class GroundingGuard:
    """Rule-based guard that verifies generated commentary against evidence."""

    QUALIFIERS: dict[str, list[str]] = {
        "high": [],            # no softening needed
        "medium": ["appears to", "seems to", "likely"],
        "low": ["possibly", "it's unclear whether", "may"],
    }

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check(
        self,
        commentary: GeneratedCommentary,
        state: RaceStateSnapshot,
        retrieval: RetrievalResult | None = None,
    ) -> GuardedCommentary:
        """Verify claims in commentary against state and retrieval evidence."""
        text = commentary.generated_text
        claims = self._extract_and_verify(text, state, retrieval)
        confidence_band = self._compute_confidence_band(claims)
        adjusted_text = self._soften_text(text, claims, confidence_band)

        return GuardedCommentary(
            timestamp=datetime.now(timezone.utc),
            source_commentary_id=commentary.event_id,
            original_text=text,
            adjusted_text=adjusted_text,
            claims=claims,
            confidence_band=confidence_band,
        )

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _extract_and_verify(
        self,
        text: str,
        state: RaceStateSnapshot,
        retrieval: RetrievalResult | None,
    ) -> list[ClaimSupport]:
        claims: list[ClaimSupport] = []
        driver_map = {d.driver_id: d for d in state.drivers}

        # --- position claims ---
        for m in _POSITION_RE.finditer(text):
            p_num = m.group(1)
            p_word = m.group(2)
            pos = int(p_num) if p_num else _ORDINAL_MAP.get(p_word.lower())
            if pos is None:
                continue
            claim_text = m.group(0)
            # Try to find which driver this refers to (look backwards for a code)
            ctx = text[max(0, m.start() - 30): m.end()]
            driver = self._find_driver_in_context(ctx, driver_map)
            if driver and driver_map[driver].position == pos:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="observed",
                    confidence=1.0,
                    evidence=f"{driver} confirmed at P{pos}",
                ))
            elif driver:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="unsupported",
                    confidence=0.1,
                    evidence=f"{driver} is actually P{driver_map[driver].position}",
                ))
            else:
                # Check if any driver is at that position
                found = any(d.position == pos for d in state.drivers)
                if found:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="observed",
                        confidence=0.8,
                        evidence=f"Position P{pos} exists in state",
                    ))
                else:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="unsupported",
                        confidence=0.2,
                    ))

        # --- gap claims ---
        for m in _GAP_RE.finditer(text):
            gap_val = float(m.group(1))
            claim_text = m.group(0)
            ctx = text[max(0, m.start() - 40): m.end()]
            driver = self._find_driver_in_context(ctx, driver_map)
            if driver and driver_map[driver].gap_ahead is not None:
                actual = driver_map[driver].gap_ahead
                if abs(actual - gap_val) < 0.5:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="observed",
                        confidence=0.9,
                        evidence=f"{driver} gap_ahead={actual:.1f}s",
                    ))
                else:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="unsupported",
                        confidence=0.2,
                        evidence=f"{driver} gap_ahead={actual:.1f}s (mismatch)",
                    ))
            elif gap_val in state.gaps.values():
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="observed",
                    confidence=0.8,
                    evidence="Gap value found in state.gaps",
                ))
            else:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="unsupported",
                    confidence=0.2,
                ))

        # --- tire claims ---
        for m in _TIRE_RE.finditer(text):
            raw_compound = m.group(1).upper()
            if raw_compound.startswith("INTER"):
                raw_compound = "INTER"
            claim_text = m.group(0)
            ctx = text[max(0, m.start() - 40): m.end()]
            driver = self._find_driver_in_context(ctx, driver_map)
            if driver and driver_map[driver].tire_compound:
                actual = driver_map[driver].tire_compound.upper()
                if actual == raw_compound:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="observed",
                        confidence=1.0,
                        evidence=f"{driver} on {actual}",
                    ))
                else:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="unsupported",
                        confidence=0.1,
                        evidence=f"{driver} actually on {actual}",
                    ))
            else:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="inferred",
                    confidence=0.5,
                ))

        # --- tire age claims ---
        for m in _TIRE_AGE_RE.finditer(text):
            age_val = int(m.group(1))
            claim_text = m.group(0)
            ctx = text[max(0, m.start() - 40): m.end()]
            driver = self._find_driver_in_context(ctx, driver_map)
            if driver and driver_map[driver].tire_age is not None:
                actual = driver_map[driver].tire_age
                if abs(actual - age_val) <= 1:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="observed",
                        confidence=0.9,
                        evidence=f"{driver} tire_age={actual}",
                    ))
                else:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="unsupported",
                        confidence=0.2,
                        evidence=f"{driver} tire_age={actual} (mismatch)",
                    ))
            else:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="unsupported",
                    confidence=0.3,
                ))

        # --- lap claims ---
        for m in _LAP_RE.finditer(text):
            lap_val = int(m.group(1))
            claim_text = m.group(0)
            if lap_val == state.lap:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="observed",
                    confidence=1.0,
                    evidence=f"Current lap is {state.lap}",
                ))
            else:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="unsupported",
                    confidence=0.3,
                ))

        # --- "closing the gap" / pace delta claims ---
        closing_re = re.compile(r"closing\s+(?:the\s+)?gap", re.IGNORECASE)
        for m in closing_re.finditer(text):
            claim_text = m.group(0)
            ctx = text[max(0, m.start() - 40): m.end()]
            driver = self._find_driver_in_context(ctx, driver_map)
            if driver and driver in state.pace_deltas:
                delta = state.pace_deltas[driver]
                if delta < 0:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="inferred",
                        confidence=0.8,
                        evidence=f"{driver} pace_delta={delta:.2f}s (negative = closing)",
                    ))
                else:
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="unsupported",
                        confidence=0.2,
                        evidence=f"{driver} pace_delta={delta:.2f}s (not closing)",
                    ))
            else:
                claims.append(ClaimSupport(
                    claim=claim_text, support_type="unsupported",
                    confidence=0.3,
                ))

        # --- predictive / assertive claims ("will overtake", "will win", etc.) ---
        for m in _ASSERTIVE_RE.finditer(text):
            claim_text = m.group(0)
            # Check retrieval evidence
            if retrieval and not retrieval.gated_out and retrieval.chunks:
                chunk_texts = " ".join(
                    c.get("text", "") for c in retrieval.chunks
                ).lower()
                if any(
                    kw in chunk_texts
                    for kw in [claim_text.lower(), "overtake", "win", "pit"]
                ):
                    claims.append(ClaimSupport(
                        claim=claim_text, support_type="retrieved",
                        confidence=0.6,
                        evidence="Retrieval chunks mention related topic",
                    ))
                    continue
            claims.append(ClaimSupport(
                claim=claim_text, support_type="unsupported",
                confidence=0.1,
            ))

        return claims

    # ------------------------------------------------------------------
    # Confidence band
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence_band(claims: list[ClaimSupport]) -> str:
        """Determine confidence band from claims."""
        if not claims:
            return "medium"
        supported_types = {"observed", "inferred", "retrieved"}
        total = len(claims)
        supported = sum(1 for c in claims if c.support_type in supported_types)
        unsupported = total - supported

        if unsupported == 0:
            return "high"
        if unsupported > total / 2:
            return "low"
        return "medium"

    # ------------------------------------------------------------------
    # Text softening
    # ------------------------------------------------------------------

    def _soften_text(
        self,
        text: str,
        claims: list[ClaimSupport],
        confidence_band: str,
    ) -> str:
        """Replace assertive language for unsupported or low-confidence claims."""
        adjusted = text
        qualifiers = self.QUALIFIERS.get(confidence_band, self.QUALIFIERS["medium"])
        if not qualifiers:
            return adjusted

        for claim in claims:
            if claim.support_type == "unsupported" or (
                claim.support_type == "inferred"
                and claim.confidence < self.config.softening_threshold
            ):
                # Soften assertive patterns in the claim's text
                for am in _ASSERTIVE_RE.finditer(adjusted):
                    assertive = am.group(0)
                    replacement = f"{qualifiers[0]} {assertive.lower().replace('will ', '')}"
                    adjusted = adjusted.replace(assertive, replacement, 1)
                    break  # one replacement per claim

        return adjusted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_driver_in_context(
        ctx: str,
        driver_map: dict[str, object],
    ) -> Optional[str]:
        """Find the nearest (rightmost) recognised driver code in context."""
        last_found: Optional[str] = None
        for m in _DRIVER_CODE_RE.finditer(ctx):
            code = m.group(1)
            if code in driver_map:
                last_found = code
        return last_found
