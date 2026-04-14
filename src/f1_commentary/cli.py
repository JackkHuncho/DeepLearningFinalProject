"""Typer CLI entry-point for the F1 commentary system.

Command groups are organised by project phase. Stubs print a message
indicating which phase will implement the command.
"""

from pathlib import Path
from typing import Optional

import typer

from f1_commentary.config import AppConfig

app = typer.Typer(
    name="f1-commentary",
    help="F1 Commentary Generation System CLI",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Phase 2 — Event bus / log replay  (sub-app)
# ---------------------------------------------------------------------------

replay_log_app = typer.Typer(
    name="replay-log",
    help="Replay and inspect persisted event logs (Phase 2).",
    no_args_is_help=True,
)
app.add_typer(replay_log_app, name="replay-log")


@replay_log_app.command()
def summary(
    log_file: Optional[Path] = typer.Argument(  # noqa: UP007
        None, help="Path to a JSONL event log. Defaults to <log_dir>/events.jsonl."
    ),
) -> None:
    """Read a JSONL event log and print event-type counts."""
    from f1_commentary.bus.log_writer import JSONLLogWriter

    cfg = AppConfig()
    path = log_file or (cfg.log_dir / "events.jsonl")
    if not path.exists():
        typer.echo(f"Log file not found: {path}")
        raise typer.Exit(code=1)

    writer = JSONLLogWriter(path)
    events = writer.read_all()
    if not events:
        typer.echo("Log is empty.")
        raise typer.Exit()

    counts: dict[str, int] = {}
    for e in events:
        etype = e.get("event_type", "unknown")
        counts[etype] = counts.get(etype, 0) + 1

    typer.echo(f"Total events: {len(events)}")
    for etype, count in sorted(counts.items()):
        typer.echo(f"  {etype}: {count}")


@replay_log_app.command()
def rebuild_test(
    log_file: Optional[Path] = typer.Argument(  # noqa: UP007
        None, help="Path to a JSONL event log. Defaults to <log_dir>/events.jsonl."
    ),
) -> None:
    """Run StateRebuildConsumer on a log file and report success/failure."""
    from f1_commentary.bus.state_rebuild import StateRebuildConsumer

    cfg = AppConfig()
    path = log_file or (cfg.log_dir / "events.jsonl")
    if not path.exists():
        typer.echo(f"Log file not found: {path}")
        raise typer.Exit(code=1)

    consumer = StateRebuildConsumer()
    state = consumer.rebuild(path)

    if state.get("success"):
        typer.echo("Rebuild succeeded.")
        typer.echo(f"  Total events: {state['total_events']}")
        for etype, count in sorted(state.get("event_counts", {}).items()):
            typer.echo(f"    {etype}: {count}")
        if state.get("last_session_key"):
            typer.echo(f"  Last session: {state['last_session_key']}")
        if state.get("last_telemetry_lap") is not None:
            typer.echo(f"  Last telemetry lap: {state['last_telemetry_lap']}")
        if state.get("final_commentaries"):
            typer.echo(f"  Final commentaries: {len(state['final_commentaries'])}")
    else:
        typer.echo("Rebuild failed.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Phase 1 — Replay
# ---------------------------------------------------------------------------

@app.command()
def replay() -> None:
    """Replay a race session from FastF1 data (Phase 1)."""
    typer.echo("Not yet implemented — Phase 1")


# ---------------------------------------------------------------------------
# Phase 4 — Recognizers
# ---------------------------------------------------------------------------

@app.command()
def run_recognizers(
    log_file: Optional[Path] = typer.Argument(  # noqa: UP007
        None, help="Path to a JSONL event log. Defaults to <log_dir>/events.jsonl."
    ),
) -> None:
    """Run all registered recognizers on RaceStateSnapshot events in a JSONL log."""
    import json

    from f1_commentary.recognizers.engine import RecognitionEngine
    from f1_commentary.schemas.events import EventType, RaceStateSnapshot

    cfg = AppConfig()
    path = log_file or (cfg.log_dir / "events.jsonl")
    if not path.exists():
        typer.echo(f"Log file not found: {path}")
        raise typer.Exit(code=1)

    engine = RecognitionEngine()
    total_snapshots = 0
    total_candidates = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                typer.echo(f"  [warn] skipping malformed JSON on line {line_no}")
                continue

            if data.get("event_type") != EventType.RACE_STATE_UPDATE.value:
                continue

            snapshot = RaceStateSnapshot(**data)
            total_snapshots += 1
            candidates = engine.process(snapshot)
            total_candidates += len(candidates)

            if candidates:
                typer.echo(
                    f"Lap {snapshot.lap} | {len(candidates)} candidate(s):"
                )
                for c in candidates:
                    typer.echo(
                        f"  [{c.confidence:.2f}] {c.candidate_type.value} "
                        f"— {', '.join(c.involved_drivers)}: {c.explanation}"
                    )

    typer.echo(f"\nProcessed {total_snapshots} snapshot(s), "
               f"found {total_candidates} candidate event(s).")


# ---------------------------------------------------------------------------
# Phase 6 — Retrieval / Memory
# ---------------------------------------------------------------------------

@app.command()
def build_index(
    chunks_file: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--chunks", "-c", help="Path to a JSON file containing memory chunks."
    ),
    output_dir: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Directory to save the built index."
    ),
) -> None:
    """Build or refresh the retrieval index from memory chunks (Phase 6)."""
    cfg = AppConfig()
    chunks_path = chunks_file or (cfg.data_dir / "chunks.json")
    out = output_dir or (cfg.data_dir / "retrieval_index")

    if not chunks_path.exists():
        typer.echo(
            f"Chunks file not found: {chunks_path}\n"
            "Provide a JSON file of MemoryChunk objects with --chunks, "
            "or place chunks.json in the data directory."
        )
        raise typer.Exit(code=1)

    import json as _json

    from f1_commentary.retrieval.embeddings import EmbeddingModel
    from f1_commentary.retrieval.ingest import IndexBuilder
    from f1_commentary.schemas.memory import MemoryChunk

    with open(chunks_path) as f:
        raw = _json.load(f)
    chunks = [MemoryChunk(**c) for c in raw]
    typer.echo(f"Loaded {len(chunks)} chunks from {chunks_path}")

    model = EmbeddingModel(cfg.retrieval.embedding_model)
    builder = IndexBuilder(model, cfg.retrieval)
    index, chunks = builder.build_from_chunks(chunks)
    builder.save(index, chunks, out)
    typer.echo(f"Index built ({index.size} vectors) and saved to {out}")


@app.command()
def search_memory(
    query: str = typer.Argument(..., help="Search query text."),
    index_dir: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--index", "-i", help="Directory containing the saved index."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return."),
) -> None:
    """Search the retrieval index with a text query (Phase 6)."""
    cfg = AppConfig()
    idx_dir = index_dir or (cfg.data_dir / "retrieval_index")

    if not idx_dir.exists():
        typer.echo(
            f"Index directory not found: {idx_dir}\n"
            "Build the index first with the build-index command."
        )
        raise typer.Exit(code=1)

    from f1_commentary.retrieval.embeddings import EmbeddingModel
    from f1_commentary.retrieval.ingest import IndexBuilder

    index, chunks = IndexBuilder.load(idx_dir, cfg.retrieval)
    model = EmbeddingModel(cfg.retrieval.embedding_model)
    query_vec = model.encode_single(query)
    results = index.search(query_vec, k=top_k)

    chunk_map = {c.chunk_id: c for c in chunks}
    if not results:
        typer.echo("No results found.")
        raise typer.Exit()

    typer.echo(f"Top {len(results)} results for: {query!r}\n")
    for rank, (chunk_id, score) in enumerate(results, 1):
        chunk = chunk_map.get(chunk_id)
        text_preview = chunk.text[:120] if chunk else "(chunk not found)"
        drivers = ", ".join(chunk.involved_drivers) if chunk else ""
        typer.echo(f"  {rank}. [{score:.4f}] {chunk_id}")
        typer.echo(f"     Drivers: {drivers}")
        typer.echo(f"     {text_preview}")
        typer.echo()


# ---------------------------------------------------------------------------
# Phase 12 — Traces
# ---------------------------------------------------------------------------

@app.command()
def trace_summary(
    trace_file: Path = typer.Argument(..., help="Path to a JSONL trace file."),
) -> None:
    """Print a summary of trace logs (Phase 12)."""
    from f1_commentary.traces.trace_logger import TraceLogger
    from f1_commentary.traces.trace_utils import trace_summary as _trace_summary

    if not trace_file.exists():
        typer.echo(f"Trace file not found: {trace_file}")
        raise typer.Exit(code=1)

    logger = TraceLogger(trace_file.parent)
    traces = logger.read_traces(trace_file)
    summary = _trace_summary(traces)

    typer.echo(f"Total traces: {summary['total_traces']}")
    typer.echo(f"Avg latency: {summary['avg_latency_ms']} ms")
    typer.echo("Event type distribution:")
    for etype, count in sorted(summary["event_type_distribution"].items()):
        typer.echo(f"  {etype}: {count}")
    typer.echo("Driver mention frequency:")
    for driver, count in sorted(summary["driver_mention_frequency"].items()):
        typer.echo(f"  {driver}: {count}")
    typer.echo("Storyline counts:")
    for sid, count in sorted(summary["storyline_counts"].items()):
        typer.echo(f"  {sid}: {count}")


@app.command()
def trace_filter(
    trace_file: Path = typer.Argument(..., help="Path to a JSONL trace file."),
    driver: Optional[str] = typer.Option(None, "--driver", "-d", help="Filter by driver code."),
    event_type: Optional[str] = typer.Option(None, "--event-type", "-e", help="Filter by candidate event type."),
    storyline: Optional[str] = typer.Option(None, "--storyline", "-s", help="Filter by storyline ID."),
) -> None:
    """Filter trace logs by criteria (Phase 12)."""
    from f1_commentary.traces.trace_logger import TraceLogger
    from f1_commentary.traces.trace_utils import filter_traces

    if not trace_file.exists():
        typer.echo(f"Trace file not found: {trace_file}")
        raise typer.Exit(code=1)

    logger = TraceLogger(trace_file.parent)
    traces = logger.read_traces(trace_file)
    filtered = filter_traces(traces, driver=driver, event_type=event_type, storyline_id=storyline)

    typer.echo(f"Matched {len(filtered)} / {len(traces)} traces")
    for t in filtered:
        ctype = t.candidate_event.get("candidate_type", "unknown")
        drivers = ", ".join(t.candidate_event.get("involved_drivers", []))
        typer.echo(f"  [{t.trace_id[:8]}] {ctype} — {drivers}")


@app.command()
def trace_export(
    trace_file: Path = typer.Argument(..., help="Path to a JSONL trace file."),
    output_path: Path = typer.Argument(..., help="Output file path."),
    fmt: str = typer.Option("json", "--format", "-f", help="Export format: csv or json."),
) -> None:
    """Export trace logs to an external format (Phase 12)."""
    from f1_commentary.traces.trace_logger import TraceLogger
    from f1_commentary.traces.trace_utils import export_traces_csv, export_traces_json

    if not trace_file.exists():
        typer.echo(f"Trace file not found: {trace_file}")
        raise typer.Exit(code=1)

    logger = TraceLogger(trace_file.parent)
    traces = logger.read_traces(trace_file)

    if fmt == "csv":
        export_traces_csv(traces, output_path)
    elif fmt == "json":
        export_traces_json(traces, output_path)
    else:
        typer.echo(f"Unknown format: {fmt}. Use 'csv' or 'json'.")
        raise typer.Exit(code=1)

    typer.echo(f"Exported {len(traces)} traces to {output_path} ({fmt})")


# ---------------------------------------------------------------------------
# Phase 14 — Evaluation
# ---------------------------------------------------------------------------

@app.command()
def evaluate_run(
    traces: Path = typer.Option(..., "--traces", "-t", help="Path to a JSONL trace file."),
    references: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--references", "-r", help="Path to a text file with one reference per line."
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Path to write JSON results."
    ),
) -> None:
    """Evaluate a commentary run against reference data (Phase 14)."""
    import json as _json

    from f1_commentary.evaluation.evaluator import TraceEvaluator

    if not traces.exists():
        typer.echo(f"Trace file not found: {traces}")
        raise typer.Exit(code=1)

    evaluator = TraceEvaluator(traces)

    refs = None
    if references and references.exists():
        refs = [line.strip() for line in open(references) if line.strip()]

    results = evaluator.evaluate(references=refs)

    typer.echo(_json.dumps(results, indent=2))

    if output:
        evaluator.export_results(results, output)
        typer.echo(f"\nResults saved to {output}")


@app.command()
def compare_system_vs_baseline(
    system_traces: Path = typer.Option(
        ..., "--system-traces", help="Path to system JSONL trace file."
    ),
    baseline_traces: Path = typer.Option(
        ..., "--baseline-traces", help="Path to baseline JSONL trace file."
    ),
    references: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--references", "-r", help="Path to a text file with one reference per line."
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Path to write JSON comparison results."
    ),
) -> None:
    """Compare system output to baseline commentaries (Phase 14)."""
    import json as _json

    from f1_commentary.evaluation.evaluator import TraceEvaluator
    from f1_commentary.evaluation.compare import SystemComparison

    for label, path in [("System", system_traces), ("Baseline", baseline_traces)]:
        if not path.exists():
            typer.echo(f"{label} trace file not found: {path}")
            raise typer.Exit(code=1)

    refs = None
    if references and references.exists():
        refs = [line.strip() for line in open(references) if line.strip()]

    sys_eval = TraceEvaluator(system_traces)
    base_eval = TraceEvaluator(baseline_traces)

    sys_results = sys_eval.evaluate(references=refs)
    base_results = base_eval.evaluate(references=refs)

    comparison = SystemComparison(sys_results, base_results)
    summary = comparison.compare()

    typer.echo(_json.dumps(summary, indent=2))

    if output:
        comparison.export(output)
        typer.echo(f"\nComparison saved to {output}")


# ---------------------------------------------------------------------------
# Phase 16 — Artifacts
# ---------------------------------------------------------------------------

@app.command()
def generate_artifacts() -> None:
    """Generate final deliverable artifacts (Phase 16)."""
    typer.echo("Not yet implemented — Phase 16")


if __name__ == "__main__":
    app()
