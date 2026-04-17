"""Data quality checks and deduplication for the SFT dataset.

Checks
======
1. **Empty targets** — examples with no target text.
2. **Too-short targets** — target below a minimum word count.
3. **Too-long targets** — target above a maximum word count.
4. **Duplicate targets** — identical target_text across examples.
5. **Unmatched examples** — alignment confidence is UNMATCHED.
6. **Low-confidence alignment** — HEURISTIC-only matches.

The ``run_quality_checks`` function returns a report dict and optionally
filters out failing examples.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.telemetry_to_narrative.schemas.sft_dataset import (
    AlignmentConfidence,
    SFTExample,
)

logger = logging.getLogger(__name__)

# Defaults.
MIN_TARGET_WORDS = 5
MAX_TARGET_WORDS = 500

# Confidence ordering: higher index = higher confidence.
CONFIDENCE_RANK: dict[AlignmentConfidence, int] = {
    AlignmentConfidence.UNMATCHED: 0,
    AlignmentConfidence.HEURISTIC: 1,
    AlignmentConfidence.NEAR: 2,
    AlignmentConfidence.EXACT: 3,
}


def confidence_at_least(
    confidence: AlignmentConfidence,
    threshold: AlignmentConfidence,
) -> bool:
    """Return True if *confidence* meets or exceeds *threshold*."""
    return CONFIDENCE_RANK[confidence] >= CONFIDENCE_RANK[threshold]


def filter_by_confidence(
    examples: list[SFTExample],
    min_confidence: AlignmentConfidence,
) -> tuple[list[SFTExample], int]:
    """Keep only examples whose alignment confidence >= *min_confidence*.

    Returns (kept_examples, dropped_count).
    """
    kept = [e for e in examples if confidence_at_least(e.alignment_confidence, min_confidence)]
    return kept, len(examples) - len(kept)


def _word_count(text: str) -> int:
    return len(text.split())


def check_empty_targets(examples: list[SFTExample]) -> list[str]:
    """Return example_ids with empty target_text."""
    return [e.example_id for e in examples if not e.target_text.strip()]


def check_short_targets(examples: list[SFTExample], min_words: int = MIN_TARGET_WORDS) -> list[str]:
    """Return example_ids with target_text below min_words."""
    return [
        e.example_id for e in examples
        if e.target_text.strip() and _word_count(e.target_text) < min_words
    ]


def check_long_targets(examples: list[SFTExample], max_words: int = MAX_TARGET_WORDS) -> list[str]:
    """Return example_ids with target_text above max_words."""
    return [e.example_id for e in examples if _word_count(e.target_text) > max_words]


def check_duplicate_targets(examples: list[SFTExample]) -> list[str]:
    """Return example_ids that share a target_text with another example.

    Only non-empty targets are checked.  All copies are flagged (not just
    the second occurrence).
    """
    target_counts: Counter[str] = Counter()
    for e in examples:
        t = e.target_text.strip()
        if t:
            target_counts[t] += 1

    duplicated_texts = {t for t, c in target_counts.items() if c > 1}
    return [e.example_id for e in examples if e.target_text.strip() in duplicated_texts]


def check_unmatched(examples: list[SFTExample]) -> list[str]:
    """Return example_ids with UNMATCHED alignment confidence."""
    return [e.example_id for e in examples if e.alignment_confidence == AlignmentConfidence.UNMATCHED]


def check_low_confidence(examples: list[SFTExample]) -> list[str]:
    """Return example_ids with HEURISTIC-only alignment."""
    return [e.example_id for e in examples if e.alignment_confidence == AlignmentConfidence.HEURISTIC]


def deduplicate(examples: list[SFTExample]) -> list[SFTExample]:
    """Remove examples with duplicate target_text, keeping the first occurrence.

    Only non-empty targets are deduped.
    """
    seen: set[str] = set()
    result: list[SFTExample] = []
    for e in examples:
        t = e.target_text.strip()
        if t and t in seen:
            continue
        if t:
            seen.add(t)
        result.append(e)
    return result


def run_quality_checks(
    examples: list[SFTExample],
    min_target_words: int = MIN_TARGET_WORDS,
    max_target_words: int = MAX_TARGET_WORDS,
    remove_empty: bool = True,
    remove_duplicates: bool = True,
    remove_unmatched: bool = True,
    min_confidence: AlignmentConfidence = AlignmentConfidence.NEAR,
) -> tuple[list[SFTExample], dict]:
    """Run all quality checks and optionally filter failing examples.

    Parameters
    ----------
    examples : list[SFTExample]
        Input examples.
    min_target_words / max_target_words : int
        Thresholds for short/long target checks.
    remove_empty : bool
        If True, remove examples with empty targets.
    remove_duplicates : bool
        If True, deduplicate by target_text.
    remove_unmatched : bool
        If True, remove UNMATCHED examples.  Superseded by *min_confidence*
        when the threshold is stricter than UNMATCHED.
    min_confidence : AlignmentConfidence
        Minimum alignment confidence to keep.  Default ``NEAR``.

    Returns
    -------
    (filtered_examples, report)
        ``report`` is a dict summarising each check's findings.
    """
    # Count examples per confidence bucket before any filtering.
    confidence_counts_before: dict[str, int] = {c.value: 0 for c in AlignmentConfidence}
    for e in examples:
        confidence_counts_before[e.alignment_confidence.value] += 1

    report: dict = {
        "total_before_filtering": len(examples),
        "confidence_counts_before": confidence_counts_before,
        "min_confidence_threshold": min_confidence.value,
        "empty_targets": check_empty_targets(examples),
        "short_targets": check_short_targets(examples, min_target_words),
        "long_targets": check_long_targets(examples, max_target_words),
        "duplicate_targets": check_duplicate_targets(examples),
        "unmatched": check_unmatched(examples),
        "low_confidence": check_low_confidence(examples),
    }

    filtered = list(examples)

    # Confidence filter (subsumes remove_unmatched when threshold > UNMATCHED).
    filtered, confidence_dropped = filter_by_confidence(filtered, min_confidence)
    report["confidence_filter_dropped"] = confidence_dropped

    # Legacy unmatched removal only if threshold is UNMATCHED (i.e. allows everything).
    if remove_unmatched and min_confidence == AlignmentConfidence.UNMATCHED:
        unmatched_ids = set(report["unmatched"])
        filtered = [e for e in filtered if e.example_id not in unmatched_ids]

    if remove_empty:
        empty_ids = set(report["empty_targets"])
        filtered = [e for e in filtered if e.example_id not in empty_ids]

    if remove_duplicates:
        filtered = deduplicate(filtered)

    # Count examples per confidence bucket after filtering.
    confidence_counts_after: dict[str, int] = {c.value: 0 for c in AlignmentConfidence}
    for e in filtered:
        confidence_counts_after[e.alignment_confidence.value] += 1

    report["total_after_filtering"] = len(filtered)
    report["confidence_counts_after"] = confidence_counts_after
    report["total_output"] = len(filtered)
    report["removed"] = report["total_before_filtering"] - report["total_output"]

    # Summary counts for metadata.
    report["summary"] = {
        "total_before_filtering": report["total_before_filtering"],
        "total_after_filtering": report["total_after_filtering"],
        "min_confidence_threshold": min_confidence.value,
        "confidence_filter_dropped": confidence_dropped,
        "confidence_counts_before": confidence_counts_before,
        "confidence_counts_after": confidence_counts_after,
        "empty_target_count": len(report["empty_targets"]),
        "short_target_count": len(report["short_targets"]),
        "long_target_count": len(report["long_targets"]),
        "duplicate_target_count": len(report["duplicate_targets"]),
        "unmatched_count": len(report["unmatched"]),
        "low_confidence_count": len(report["low_confidence"]),
        "total_removed": report["removed"],
    }

    logger.info(
        "Quality checks: %d in → %d out (removed %d: "
        "%d by confidence filter [>=%s], %d empty, duplicates deduped).",
        report["total_before_filtering"], report["total_output"], report["removed"],
        confidence_dropped, min_confidence.value,
        len(report["empty_targets"]),
    )

    return filtered, report
