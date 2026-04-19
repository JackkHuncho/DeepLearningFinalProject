"""Trace utilities: replay, filter, export."""

import csv
import json
from pathlib import Path
from typing import Optional

from f1_commentary.traces.trace_logger import TraceEntry, TraceLogger


def filter_traces(
    traces: list[TraceEntry],
    driver: Optional[str] = None,
    event_type: Optional[str] = None,
    storyline_id: Optional[str] = None,
    session_key: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> list[TraceEntry]:
    """Filter traces by various criteria."""
    result = traces

    if driver is not None:
        result = [
            t for t in result
            if driver in t.candidate_event.get("involved_drivers", [])
            or driver in t.final_commentary.get("involved_drivers", [])
            or driver in t.generated_commentary.get("involved_drivers", [])
        ]

    if event_type is not None:
        result = [
            t for t in result
            if t.candidate_event.get("candidate_type") == event_type
        ]

    if storyline_id is not None:
        result = [
            t for t in result
            if t.storyline_id == storyline_id
        ]

    if session_key is not None:
        result = [
            t for t in result
            if t.telemetry_snapshot.get("session_key") == session_key
        ]

    if min_confidence is not None:
        result = [
            t for t in result
            if t.candidate_event.get("confidence", 0.0) >= min_confidence
        ]

    return result


def trace_summary(traces: list[TraceEntry]) -> dict:
    """Generate summary statistics from traces."""
    if not traces:
        return {
            "total_traces": 0,
            "event_type_distribution": {},
            "avg_latency_ms": 0.0,
            "driver_mention_frequency": {},
            "storyline_counts": {},
        }

    event_type_dist: dict[str, int] = {}
    driver_freq: dict[str, int] = {}
    storyline_counts: dict[str, int] = {}
    total_latency = 0.0

    for t in traces:
        # Event type distribution
        ctype = t.candidate_event.get("candidate_type", "unknown")
        event_type_dist[ctype] = event_type_dist.get(ctype, 0) + 1

        # Driver mention frequency
        for driver in t.candidate_event.get("involved_drivers", []):
            driver_freq[driver] = driver_freq.get(driver, 0) + 1
        for driver in t.final_commentary.get("involved_drivers", []):
            driver_freq[driver] = driver_freq.get(driver, 0) + 1

        # Storyline counts
        sid = t.storyline_id or "none"
        storyline_counts[sid] = storyline_counts.get(sid, 0) + 1

        # Latency
        total_latency += t.total_latency_ms

    avg_latency = total_latency / len(traces) if traces else 0.0

    return {
        "total_traces": len(traces),
        "event_type_distribution": event_type_dist,
        "avg_latency_ms": round(avg_latency, 2),
        "driver_mention_frequency": driver_freq,
        "storyline_counts": storyline_counts,
    }


def export_traces_csv(traces: list[TraceEntry], output_path: Path) -> None:
    """Export traces to CSV for analysis."""
    if not traces:
        output_path.write_text("")
        return

    fieldnames = [
        "trace_id",
        "beat_id",
        "event_timestamp",
        "storyline_id",
        "model_variant",
        "total_latency_ms",
        "candidate_type",
        "involved_drivers",
        "confidence",
        "final_text",
        "confidence_band",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            writer.writerow({
                "trace_id": t.trace_id,
                "beat_id": t.beat_id,
                "event_timestamp": t.event_timestamp.isoformat(),
                "storyline_id": t.storyline_id or "",
                "model_variant": t.model_variant,
                "total_latency_ms": t.total_latency_ms,
                "candidate_type": t.candidate_event.get("candidate_type", ""),
                "involved_drivers": ",".join(
                    t.candidate_event.get("involved_drivers", [])
                ),
                "confidence": t.candidate_event.get("confidence", ""),
                "final_text": t.final_commentary.get("final_text", ""),
                "confidence_band": t.final_commentary.get("confidence_band", ""),
            })


def export_traces_json(traces: list[TraceEntry], output_path: Path) -> None:
    """Export traces to a single JSON file."""
    data = [json.loads(t.model_dump_json()) for t in traces]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
