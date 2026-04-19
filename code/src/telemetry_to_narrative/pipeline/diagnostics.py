"""Phase 17 diagnostics — doctor + trace linkage validators.

Two read-only utilities used by the CLI for sanity-checking a setup or
a completed run.  Neither writes anything; both return structured dicts
that the CLI prints.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Type

from pydantic import BaseModel, ValidationError

from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.candidate_event import CandidateEvent
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.final_commentary import FinalCommentary
from src.telemetry_to_narrative.schemas.baseline_commentary import BaselineCommentary

logger = logging.getLogger(__name__)


# Stage filename → schema used for that file.  Drives both the linkage
# validator and the schema-compat checker.
TRACE_SCHEMAS: dict[str, Type[BaseModel]] = {
    "replay_frames.jsonl":        TelemetryFrame,
    "snapshots.jsonl":            RaceStateSnapshot,
    "candidate_events.jsonl":     CandidateEvent,
    "scheduled_beats.jsonl":      ScheduledBeat,
    "generated_commentary.jsonl": GeneratedCommentary,
    "final_commentary.jsonl":     FinalCommentary,
    "baseline_commentary.jsonl":  BaselineCommentary,
}


# ── Doctor ────────────────────────────────────────────────────────────

_REQUIRED_FOLDERS = ("data", "data/logs", "data/artifacts")
_OPTIONAL_DEPS = ("fastf1", "openai", "mlx", "llama_cpp")
_SAMPLE_PATHS = (
    "data/logs/sample_replay.jsonl",
    "data/artifacts/sample_run_local",
    "data/artifacts/sample_run_baseline",
)


def run_doctor(repo_root: str | Path = ".") -> dict:
    """Return a structured health report for the local checkout.

    Checks four categories: required folders, optional Python deps,
    backend reachability (mock always reachable; openai needs key),
    and presence of sample artifacts.  No remote calls are made.
    """
    root = Path(repo_root).resolve()
    report: dict = {
        "repo_root": str(root),
        "folders": {},
        "dependencies": {},
        "backends": {},
        "samples": {},
        "ok": True,
    }

    # Folders.
    for rel in _REQUIRED_FOLDERS:
        p = root / rel
        report["folders"][rel] = {"exists": p.is_dir(), "path": str(p)}
        if not p.is_dir():
            report["ok"] = False

    # Optional dependencies — installed or not.
    for dep in _OPTIONAL_DEPS:
        try:
            importlib.import_module(dep)
            report["dependencies"][dep] = "installed"
        except Exception:
            report["dependencies"][dep] = "missing"

    # Backends — mock always works; others need their dep + (for openai)
    # an API key in the environment.
    report["backends"]["local:mock"] = "ready"
    report["backends"]["local:mlx"] = (
        "ready" if report["dependencies"]["mlx"] == "installed" else "unavailable"
    )
    report["backends"]["local:llama_cpp"] = (
        "ready" if report["dependencies"]["llama_cpp"] == "installed"
        else "unavailable"
    )
    report["backends"]["baseline:mock"] = "ready"
    if report["dependencies"]["openai"] != "installed":
        report["backends"]["baseline:openai"] = "unavailable (no openai package)"
    elif not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
        report["backends"]["baseline:openai"] = "missing OPENAI_API_KEY"
    else:
        report["backends"]["baseline:openai"] = "ready"

    # Samples — used by docs/tests, helpful when missing.
    for rel in _SAMPLE_PATHS:
        p = root / rel
        report["samples"][rel] = "present" if p.exists() else "missing"

    return report


# ── Schema compatibility check ────────────────────────────────────────

def check_jsonl_schema(
    path: str | Path,
    schema_cls: Type[BaseModel],
    *,
    max_errors: int = 5,
) -> dict:
    """Validate every line of a JSONL file against ``schema_cls``.

    Returns a dict ``{path, schema, total, valid, errors}``.  Stops
    collecting error detail after ``max_errors`` to keep output small.
    """
    p = Path(path)
    result = {
        "path": str(p),
        "schema": schema_cls.__name__,
        "total": 0,
        "valid": 0,
        "errors": [],
    }
    if not p.exists():
        result["errors"].append({"line": None, "msg": "file not found"})
        return result

    with open(p) as fh:
        for i, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            result["total"] += 1
            try:
                schema_cls.model_validate_json(stripped)
                result["valid"] += 1
            except ValidationError as exc:
                if len(result["errors"]) < max_errors:
                    result["errors"].append({"line": i, "msg": str(exc)[:240]})
            except Exception as exc:  # malformed JSON
                if len(result["errors"]) < max_errors:
                    result["errors"].append(
                        {"line": i, "msg": f"parse error: {exc}"}
                    )
    return result


# ── Trace linkage validator ───────────────────────────────────────────

def _load_jsonl(path: Path, schema_cls: Type[BaseModel]) -> list[BaseModel]:
    out: list[BaseModel] = []
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(schema_cls.model_validate_json(stripped))
    return out


def validate_trace_linkage(trace_dir: str | Path) -> dict:
    """Verify ID propagation across the JSONL traces in ``trace_dir``.

    Linkage checked (only for files that exist — baseline-mode runs
    won't have local generated/finals):

      - snapshot.frame_id ⊆ frame.frame_id
      - candidate.source_frame_id ⊆ snapshot.frame_id
      - beat.source_frame_id ⊆ snapshot.frame_id
      - generation.beat_id ⊆ beat.beat_id
      - final.beat_id ⊆ beat.beat_id
      - final.source_generation_id ⊆ generation.generation_id
      - baseline.beat_id ⊆ beat.beat_id

    Returns a ``{ok, errors, counts}`` report.
    """
    base = Path(trace_dir)
    errors: list[str] = []
    counts: dict[str, int] = {}

    frames = _load_jsonl(base / "replay_frames.jsonl", TelemetryFrame)
    snapshots = _load_jsonl(base / "snapshots.jsonl", RaceStateSnapshot)
    candidates = _load_jsonl(base / "candidate_events.jsonl", CandidateEvent)
    beats = _load_jsonl(base / "scheduled_beats.jsonl", ScheduledBeat)
    generated = _load_jsonl(base / "generated_commentary.jsonl", GeneratedCommentary)
    finals = _load_jsonl(base / "final_commentary.jsonl", FinalCommentary)
    baselines = _load_jsonl(base / "baseline_commentary.jsonl", BaselineCommentary)

    counts.update({
        "frames": len(frames),
        "snapshots": len(snapshots),
        "candidates": len(candidates),
        "beats": len(beats),
        "generated": len(generated),
        "finals": len(finals),
        "baselines": len(baselines),
    })

    frame_ids = {f.frame_id for f in frames}
    snap_frame_ids = {s.frame_id for s in snapshots}
    beat_ids = {b.beat_id for b in beats}
    gen_ids = {g.generation_id for g in generated}

    # snapshot.frame_id ⊆ frame_ids (only meaningful when both populated).
    if frame_ids and snap_frame_ids:
        missing = snap_frame_ids - frame_ids
        if missing:
            errors.append(
                f"{len(missing)} snapshot frame_ids not present in frames "
                f"(sample: {sorted(missing)[:3]})"
            )

    if snap_frame_ids:
        cand_orphans = {c.source_frame_id for c in candidates} - snap_frame_ids
        if cand_orphans:
            errors.append(
                f"{len(cand_orphans)} candidate.source_frame_id not in snapshots"
            )
        beat_orphans = {b.source_frame_id for b in beats} - snap_frame_ids
        if beat_orphans:
            errors.append(
                f"{len(beat_orphans)} beat.source_frame_id not in snapshots"
            )

    if beat_ids:
        gen_orphans = {g.beat_id for g in generated} - beat_ids
        if gen_orphans:
            errors.append(
                f"{len(gen_orphans)} generation.beat_id not in beats"
            )
        final_beat_orphans = {f.beat_id for f in finals} - beat_ids
        if final_beat_orphans:
            errors.append(
                f"{len(final_beat_orphans)} final.beat_id not in beats"
            )
        baseline_orphans = {b.beat_id for b in baselines} - beat_ids
        if baseline_orphans:
            errors.append(
                f"{len(baseline_orphans)} baseline.beat_id not in beats"
            )

    # final.source_generation_id ⊆ gen_ids (allow None — fallback finals
    # have no source generation).
    if gen_ids:
        missing_src = {
            f.source_generation_id for f in finals
            if f.source_generation_id is not None
        } - gen_ids
        if missing_src:
            errors.append(
                f"{len(missing_src)} final.source_generation_id not in generated"
            )

    return {"ok": not errors, "errors": errors, "counts": counts}


def check_trace_schemas(trace_dir: str | Path) -> dict:
    """Run :func:`check_jsonl_schema` over every known stage file."""
    base = Path(trace_dir)
    per_file: dict[str, dict] = {}
    ok = True
    for fname, schema_cls in TRACE_SCHEMAS.items():
        path = base / fname
        if not path.exists():
            per_file[fname] = {"skipped": "file not present"}
            continue
        report = check_jsonl_schema(path, schema_cls)
        per_file[fname] = report
        if report["errors"]:
            ok = False
    return {"ok": ok, "files": per_file}


def load_run_summary(trace_dir: str | Path) -> dict | None:
    """Return the parsed run_summary.json for a trace directory, or None."""
    p = Path(trace_dir) / "run_summary.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)
