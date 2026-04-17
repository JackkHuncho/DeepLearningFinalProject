"""CLI for the telemetry-to-narrative replay system."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from src.telemetry_to_narrative.adapters.session_loader import load_session
from src.telemetry_to_narrative.adapters.replay_adapter import ReplayAdapter
from src.telemetry_to_narrative.schemas.telemetry_frame import TelemetryFrame
from src.telemetry_to_narrative.state.race_state_engine import RaceStateEngine
from src.telemetry_to_narrative.state.feature_extractor import FeatureExtractor
from src.telemetry_to_narrative.state.inspection import (
    print_snapshot,
    save_snapshots_jsonl,
    save_snapshots_csv,
    compare_snapshots,
)
from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.scheduler.editorial_scheduler import EditorialScheduler
from src.telemetry_to_narrative.scheduler.snapshot_to_candidates import extract_candidates
from src.telemetry_to_narrative.config.scheduler_config import SchedulerConfig
from src.telemetry_to_narrative.training.dataset_builder import build_dataset
from src.telemetry_to_narrative.training.transcript_loader import load_transcript
from src.telemetry_to_narrative.schemas.sft_dataset import AlignmentConfidence
from src.telemetry_to_narrative.schemas.generated_commentary import GenerationConfig, GeneratedCommentary
from src.telemetry_to_narrative.generation.model_loader import load_backend
from src.telemetry_to_narrative.generation.commentary_generator import CommentaryGenerator
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.beat_manager.beat_manager import BeatManager
from src.telemetry_to_narrative.beat_manager.output_adapters import TerminalAdapter, SSEAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_speed(speed_str: str) -> float:
    """Parse speed strings like '1x', '5x', '10x' into a float multiplier."""
    cleaned = speed_str.lower().strip().rstrip("x")
    try:
        return float(cleaned)
    except ValueError:
        raise click.BadParameter(f"Invalid speed: '{speed_str}'. Use e.g. 1x, 5x, 10x")


@click.group()
def cli():
    """Telemetry-to-Narrative replay system."""
    pass


@cli.command()
@click.option("--year", type=int, required=True, help="Season year (e.g. 2024)")
@click.option("--gp", type=str, required=True, help="Grand Prix name (e.g. Monza)")
@click.option(
    "--session",
    "session_type",
    type=click.Choice(["FP1", "FP2", "FP3", "Q", "R", "S", "SQ"], case_sensitive=False),
    required=True,
    help="Session type",
)
@click.option("--speed", default="1x", help="Replay speed (e.g. 1x, 5x, 10x)")
@click.option("--max-frames", type=int, default=None, help="Limit number of frames")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Path for JSONL output. If omitted, frames are printed to stdout.",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="Override FastF1 cache directory.",
)
def replay(
    year: int,
    gp: str,
    session_type: str,
    speed: str,
    max_frames: int | None,
    output: str | None,
    cache_dir: str | None,
):
    """Replay a historical F1 session as timestamped telemetry frames."""
    speed_mult = _parse_speed(speed)

    click.echo(f"Loading {year} {gp} {session_type} ...")
    session = load_session(year, gp, session_type, cache_dir=cache_dir)

    adapter = ReplayAdapter(session, speed=speed_mult)
    click.echo(f"Session loaded: {adapter.frame_count} frames available.")

    if output:
        # JSONL file mode
        out_path = adapter.to_jsonl(output, max_frames=max_frames)
        click.echo(f"Wrote frames to {out_path}")
    else:
        # Generator / stdout mode
        for frame in adapter.replay(max_frames=max_frames, realtime=(speed_mult <= 1.0)):
            click.echo(frame.model_dump_json())


def _load_frames_from_jsonl(path: str, max_frames: int | None) -> list[TelemetryFrame]:
    """Read TelemetryFrame objects from a JSONL file."""
    frames: list[TelemetryFrame] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if max_frames is not None and i >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            frames.append(TelemetryFrame.model_validate_json(line))
    return frames


def _run_engine(frames: list[TelemetryFrame]):
    """Run the RaceStateEngine + FeatureExtractor over frames, return snapshots."""
    from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot

    engine = RaceStateEngine()
    extractor = FeatureExtractor(engine)
    snapshots = []
    for frame in frames:
        engine.update(frame)
        snap = extractor.extract(frame)
        snapshots.append(snap)
    return snapshots


@cli.command("build-state")
@click.option(
    "--from-jsonl",
    "jsonl_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a TelemetryFrame JSONL file (output of replay command).",
)
@click.option("--max-frames", type=int, default=None, help="Limit number of frames")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for snapshot JSONL (default: print to stdout).",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(),
    default=None,
    help="Also write a flattened CSV to this path.",
)
def build_state(
    jsonl_path: str,
    max_frames: int | None,
    output: str | None,
    csv_path: str | None,
):
    """Build RaceStateSnapshots from a TelemetryFrame JSONL file."""
    click.echo(f"Reading frames from {jsonl_path} ...")
    frames = _load_frames_from_jsonl(jsonl_path, max_frames)
    click.echo(f"Loaded {len(frames)} frames.")

    snapshots = _run_engine(frames)
    click.echo(f"Built {len(snapshots)} snapshots.")

    if output:
        save_snapshots_jsonl(snapshots, output)
        click.echo(f"Wrote snapshots to {output}")

    if csv_path:
        save_snapshots_csv(snapshots, csv_path)
        click.echo(f"Wrote CSV to {csv_path}")

    if not output and not csv_path:
        for snap in snapshots:
            click.echo(snap.model_dump_json())


@cli.command("inspect-state")
@click.option(
    "--from-jsonl",
    "jsonl_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a TelemetryFrame JSONL file.",
)
@click.option("--max-frames", type=int, default=None, help="Limit number of frames")
@click.option(
    "--driver",
    "drivers",
    type=str,
    multiple=True,
    help="Driver abbreviation(s) to inspect (e.g. --driver VER --driver HAM). Omit for all.",
)
@click.option(
    "--diff/--no-diff",
    default=False,
    help="Show frame-to-frame diffs.",
)
def inspect_state(
    jsonl_path: str,
    max_frames: int | None,
    drivers: tuple[str, ...],
    diff: bool,
):
    """Inspect RaceStateSnapshots for specific drivers, with optional diffs."""
    frames = _load_frames_from_jsonl(jsonl_path, max_frames)
    snapshots = _run_engine(frames)

    driver_list = list(drivers) if drivers else None

    prev = None
    for snap in snapshots:
        click.echo(print_snapshot(snap, drivers=driver_list))

        if diff and prev is not None:
            delta = compare_snapshots(prev, snap)
            if delta["position_changes"] or delta["new_battles"] or delta["ended_battles"] or delta["pit_stops"] or delta["flag_change"]:
                click.echo(f"  DIFF: {json.dumps(delta, indent=2)}")

        click.echo("")
        prev = snap


def _load_snapshots_from_jsonl(path: str, max_frames: int | None) -> list[RaceStateSnapshot]:
    """Read RaceStateSnapshot objects from a JSONL file."""
    snapshots: list[RaceStateSnapshot] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if max_frames is not None and i >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            snapshots.append(RaceStateSnapshot.model_validate_json(line))
    return snapshots


@cli.command("schedule-demo")
@click.option(
    "--from-jsonl",
    "jsonl_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a RaceStateSnapshot JSONL file.",
)
@click.option("--max-frames", type=int, default=None, help="Limit number of frames")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for scheduled beats JSONL.",
)
@click.option(
    "--queue-depth",
    type=int,
    default=0,
    help="Simulated queue depth for backpressure testing.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Override narration threshold.",
)
def schedule_demo(
    jsonl_path: str,
    max_frames: int | None,
    output: str | None,
    queue_depth: int,
    threshold: float | None,
):
    """Run the EditorialScheduler on RaceStateSnapshot JSONL and show decisions."""
    snapshots = _load_snapshots_from_jsonl(jsonl_path, max_frames)
    click.echo(f"Loaded {len(snapshots)} snapshots.")

    config = SchedulerConfig()
    if threshold is not None:
        config.narration_threshold = threshold

    scheduler = EditorialScheduler(config)
    scheduler.queue_depth = queue_depth

    all_beats = []
    prev_snap = None
    for snap in snapshots:
        candidates = extract_candidates(snap, prev_snap)
        safety_car = bool(snap.global_state.safety_car)
        beats = scheduler.schedule(candidates, frame_id=snap.frame_id, safety_car_active=safety_car)

        for beat in beats:
            marker = ">>>" if beat.selected_for_generation else "   "
            supp = f" [{beat.suppression_reason.value}]" if not beat.selected_for_generation else ""
            click.echo(
                f"  {marker} F{beat.source_frame_id:>3}  "
                f"score={beat.priority_score:.3f}  "
                f"{beat.event_type.value:<20}  "
                f"drivers={','.join(beat.involved_drivers):<12}  "
                f"story={beat.storyline_id or '':<25}  "
                f"ret={'Y' if beat.retrieval_allowed else 'N'}  "
                f"budget={beat.output_length_budget}w"
                f"{supp}"
            )

        all_beats.extend(beats)
        prev_snap = snap

    # Print counters.
    click.echo("\n--- Scheduler Counters ---")
    for k, v in scheduler.counters.as_dict().items():
        click.echo(f"  {k}: {v}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for beat in all_beats:
                fh.write(beat.model_dump_json() + "\n")
        click.echo(f"\nWrote {len(all_beats)} beats to {output}")


@cli.command("schedule-inspect")
@click.option(
    "--from-jsonl",
    "jsonl_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a RaceStateSnapshot JSONL file.",
)
@click.option("--frame", type=int, required=True, help="Frame ID to inspect.")
def schedule_inspect(jsonl_path: str, frame: int):
    """Show detailed scheduling decisions for a single frame."""
    snapshots = _load_snapshots_from_jsonl(jsonl_path, max_frames=frame + 1)
    if frame >= len(snapshots):
        click.echo(f"Frame {frame} not found (file has {len(snapshots)} snapshots).")
        return

    scheduler = EditorialScheduler()

    prev_snap = None
    target_beats = None
    for snap in snapshots:
        candidates = extract_candidates(snap, prev_snap)
        safety_car = bool(snap.global_state.safety_car)
        beats = scheduler.schedule(candidates, frame_id=snap.frame_id, safety_car_active=safety_car)
        if snap.frame_id == frame:
            target_beats = beats
        prev_snap = snap

    if target_beats is None:
        click.echo(f"No beats found for frame {frame}.")
        return

    click.echo(f"=== Frame {frame} — {len(target_beats)} beats ===\n")
    for beat in target_beats:
        sel = "SELECTED" if beat.selected_for_generation else f"SUPPRESSED ({beat.suppression_reason.value})"
        click.echo(f"  Beat: {beat.beat_id}")
        click.echo(f"    Type:       {beat.event_type.value}")
        click.echo(f"    Drivers:    {', '.join(beat.involved_drivers)}")
        click.echo(f"    Storyline:  {beat.storyline_id}")
        click.echo(f"    Decision:   {sel}")
        click.echo(f"    Score:      {beat.priority_score:.4f}")
        sb = beat.score_breakdown
        click.echo(f"      ctx_importance = {sb.contextual_importance:.4f}")
        click.echo(f"      urgency        = {sb.urgency:.4f}")
        click.echo(f"      race_phase     = {sb.race_phase:.4f}")
        click.echo(f"      story_rel      = {sb.storyline_relevance:.4f}")
        click.echo(f"      confidence     = {sb.confidence:.4f}")
        click.echo(f"      bp_penalty     = {sb.backpressure_penalty:.4f}")
        click.echo(f"    Retrieval:  {'allowed' if beat.retrieval_allowed else 'DISABLED'}")
        click.echo(f"    Budget:     {beat.output_length_budget} words")
        if beat.description:
            click.echo(f"    Desc:       {beat.description}")
        click.echo("")


@cli.command("build-sft-dataset")
@click.option(
    "--snapshots",
    "snapshot_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a RaceStateSnapshot JSONL file.",
)
@click.option(
    "--beats",
    "beats_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a ScheduledBeat JSONL file.",
)
@click.option(
    "--transcript",
    "transcript_paths",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Path(s) to transcript files (JSON, JSONL, CSV, or TXT).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/datasets",
    help="Output directory for dataset files.",
)
@click.option(
    "--alignment-mode",
    type=click.Choice(["strict", "loose"], case_sensitive=False),
    default="loose",
    help="Transcript alignment mode.",
)
@click.option("--tolerance", type=float, default=10.0, help="Near-match tolerance in seconds.")
@click.option("--val-fraction", type=float, default=0.2, help="Validation split fraction.")
@click.option("--seed", type=int, default=42, help="Random seed for split.")
@click.option("--context-radius", type=int, default=2, help="Context window radius in frames.")
@click.option(
    "--min-confidence",
    type=click.Choice(["exact", "near", "heuristic", "unmatched"], case_sensitive=False),
    default="near",
    help="Minimum alignment confidence to keep (default: near). "
         "Ordering: exact > near > heuristic > unmatched.",
)
def build_sft_dataset(
    snapshot_path: str,
    beats_path: str,
    transcript_paths: tuple[str, ...],
    output_dir: str,
    alignment_mode: str,
    tolerance: float,
    val_fraction: float,
    seed: int,
    context_radius: int,
    min_confidence: str,
):
    """Build an SFT dataset from snapshots, beats, and transcripts."""
    confidence_threshold = AlignmentConfidence(min_confidence.lower())

    click.echo(f"Building SFT dataset...")
    click.echo(f"  Snapshots:       {snapshot_path}")
    click.echo(f"  Beats:           {beats_path}")
    click.echo(f"  Transcripts:     {', '.join(transcript_paths)}")
    click.echo(f"  Output:          {output_dir}")
    click.echo(f"  Alignment:       {alignment_mode} (tolerance={tolerance}s)")
    click.echo(f"  Min confidence:  {confidence_threshold.value}")

    metadata = build_dataset(
        snapshot_path=snapshot_path,
        beats_path=beats_path,
        transcript_paths=list(transcript_paths),
        output_dir=output_dir,
        context_radius=context_radius,
        alignment_mode=alignment_mode,
        tolerance_sec=tolerance,
        val_fraction=val_fraction,
        split_seed=seed,
        min_confidence=confidence_threshold,
    )

    click.echo(f"\n--- Dataset Summary ---")
    click.echo(f"  Min confidence:  {metadata.quality_report.get('min_confidence_threshold', 'n/a')}")
    click.echo(f"  Before filter:   {metadata.quality_report.get('total_before_filtering', 'n/a')}")
    click.echo(f"  Confidence drop: {metadata.quality_report.get('confidence_filter_dropped', 0)}")
    click.echo(f"  After filter:    {metadata.total_examples}")
    click.echo(f"  Train:           {metadata.train_examples}")
    click.echo(f"  Val:             {metadata.val_examples}")
    click.echo(f"  Sessions:        {len(metadata.sessions)}")
    click.echo(f"  Event types:     {dict(metadata.event_type_distribution)}")
    click.echo(f"\n  Confidence distribution (after filter):")
    for conf in ("exact", "near", "heuristic", "unmatched"):
        count = dict(metadata.alignment_distribution).get(conf, 0)
        click.echo(f"    {conf:<15} {count}")
    click.echo(f"\n  Quality:         {metadata.quality_report}")


@cli.command("inspect-sft-dataset")
@click.option(
    "--from-jsonl",
    "jsonl_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to an SFT dataset JSONL file (train or val).",
)
@click.option("--max-examples", type=int, default=5, help="Max examples to display.")
@click.option("--show-input/--no-input", default=True, help="Show full input_text.")
@click.option("--show-target/--no-target", default=True, help="Show full target_text.")
def inspect_sft_dataset(
    jsonl_path: str,
    max_examples: int,
    show_input: bool,
    show_target: bool,
):
    """Inspect individual SFT examples from a dataset file."""
    from src.telemetry_to_narrative.schemas.sft_dataset import SFTExample
    from collections import Counter

    all_examples: list[SFTExample] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_examples.append(SFTExample.model_validate_json(line))

    # Show confidence summary header.
    conf_counts: Counter[str] = Counter()
    for ex in all_examples:
        conf_counts[ex.alignment_confidence.value] += 1

    click.echo(f"File: {jsonl_path}  ({len(all_examples)} examples)")
    click.echo(f"Alignment confidence breakdown:")
    for conf in ("exact", "near", "heuristic", "unmatched"):
        click.echo(f"  {conf:<15} {conf_counts.get(conf, 0)}")
    click.echo("")

    display = all_examples[:max_examples] if max_examples else all_examples
    click.echo(f"Showing {len(display)} of {len(all_examples)} examples:\n")

    for ex in display:
        conf_label = ex.alignment_confidence.value.upper()
        click.echo(f"=== Example: {ex.example_id}  [{conf_label}] ===")
        click.echo(f"  Event:       {ex.event_type.value}")
        click.echo(f"  Drivers:     {', '.join(ex.involved_drivers)}")
        click.echo(f"  Storyline:   {ex.storyline_id or '(none)'}")
        click.echo(f"  Alignment:   {ex.alignment_confidence.value}")
        click.echo(f"  Facts:       {len(ex.observed_facts)}")
        click.echo(f"  Inferences:  {len(ex.derived_inferences)}")

        if show_input:
            click.echo(f"\n  --- Input Text ---")
            for line in ex.input_text.split("\n"):
                click.echo(f"  {line}")

        if show_target:
            click.echo(f"\n  --- Target Text ---")
            click.echo(f"  {ex.target_text}")

        click.echo("")


@cli.command("dataset-stats")
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing sft_train.jsonl, sft_val.jsonl, sft_metadata.json.",
)
def dataset_stats(dataset_dir: str):
    """Show statistics for a generated SFT dataset."""
    from src.telemetry_to_narrative.schemas.sft_dataset import DatasetMetadata, SFTExample
    from collections import Counter

    d = Path(dataset_dir)
    meta_path = d / "sft_metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            metadata = DatasetMetadata.model_validate_json(fh.read())
        qr = metadata.quality_report

        click.echo("=== Dataset Metadata ===")
        click.echo(f"  Created:         {metadata.created_at}")
        click.echo(f"  Pipeline:        {metadata.pipeline_version}")
        click.echo(f"  Split seed:      {metadata.split_seed}")
        click.echo(f"  Val fraction:    {metadata.val_fraction}")
        click.echo(f"  Transcript sources: {', '.join(metadata.transcript_sources)}")

        click.echo(f"\n=== Confidence Filtering ===")
        click.echo(f"  Threshold:             {qr.get('min_confidence_threshold', 'n/a')}")
        click.echo(f"  Before filter:         {qr.get('total_before_filtering', 'n/a')}")
        click.echo(f"  Dropped by confidence: {qr.get('confidence_filter_dropped', 0)}")
        click.echo(f"  After filter:          {metadata.total_examples}")

        # Before-filter counts.
        before_counts = qr.get("confidence_counts_before", {})
        if before_counts:
            click.echo(f"\n  Confidence buckets (before filter):")
            for conf in ("exact", "near", "heuristic", "unmatched"):
                click.echo(f"    {conf:<15} {before_counts.get(conf, 0)}")

        # After-filter counts.
        after_counts = qr.get("confidence_counts_after", {})
        if after_counts:
            click.echo(f"\n  Confidence buckets (after filter):")
            for conf in ("exact", "near", "heuristic", "unmatched"):
                click.echo(f"    {conf:<15} {after_counts.get(conf, 0)}")

        click.echo(f"\n=== Dataset Size ===")
        click.echo(f"  Total examples:  {metadata.total_examples}")
        click.echo(f"  Train / Val:     {metadata.train_examples} / {metadata.val_examples}")

        click.echo(f"\n  Event type distribution:")
        for etype, count in sorted(metadata.event_type_distribution.items()):
            click.echo(f"    {etype:<25} {count}")

        click.echo(f"\n  Alignment distribution (final):")
        for conf in ("exact", "near", "heuristic", "unmatched"):
            count = dict(metadata.alignment_distribution).get(conf, 0)
            click.echo(f"    {conf:<15} {count}")

        click.echo(f"\n=== Quality Report ===")
        click.echo(f"  Empty targets:   {qr.get('empty_target_count', 0)}")
        click.echo(f"  Short targets:   {qr.get('short_target_count', 0)}")
        click.echo(f"  Long targets:    {qr.get('long_target_count', 0)}")
        click.echo(f"  Duplicates:      {qr.get('duplicate_target_count', 0)}")
        click.echo(f"  Total removed:   {qr.get('total_removed', 0)}")
    else:
        click.echo(f"No metadata file found at {meta_path}")

    # Compute additional stats from actual data files.
    for split_name in ("train", "val"):
        split_path = d / f"sft_{split_name}.jsonl"
        if not split_path.exists():
            continue
        examples = []
        with open(split_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(SFTExample.model_validate_json(line))

        if not examples:
            continue

        # Per-split confidence breakdown.
        conf_counts: Counter[str] = Counter()
        for ex in examples:
            conf_counts[ex.alignment_confidence.value] += 1

        word_counts = [len(ex.target_text.split()) for ex in examples]
        avg_words = sum(word_counts) / len(word_counts)
        click.echo(f"\n  {split_name.upper()} split ({len(examples)} examples):")
        click.echo(f"    Avg target words:  {avg_words:.1f}")
        click.echo(f"    Min target words:  {min(word_counts)}")
        click.echo(f"    Max target words:  {max(word_counts)}")
        click.echo(f"    Confidence: " + ", ".join(
            f"{c}={conf_counts.get(c, 0)}" for c in ("exact", "near", "heuristic", "unmatched")
        ))


def _load_beats_from_jsonl(path: str, max_items: int | None = None) -> list[ScheduledBeat]:
    """Read ScheduledBeat objects from a JSONL file."""
    beats: list[ScheduledBeat] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if max_items is not None and i >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            beats.append(ScheduledBeat.model_validate_json(line))
    return beats


@cli.command("generate-commentary")
@click.option(
    "--snapshots",
    "snapshot_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a RaceStateSnapshot JSONL file.",
)
@click.option(
    "--beats",
    "beats_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a ScheduledBeat JSONL file.",
)
@click.option(
    "--backend",
    type=click.Choice(["mock", "mlx", "llama_cpp"], case_sensitive=False),
    default="mock",
    help="Generator backend to use.",
)
@click.option("--model-path", type=click.Path(), default=None, help="Path to base model.")
@click.option("--adapter-path", type=click.Path(), default=None, help="Path to SFT adapter.")
@click.option("--max-tokens", type=int, default=128, help="Max new tokens to generate.")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature.")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for generated commentary JSONL.",
)
@click.option("--max-items", type=int, default=None, help="Limit number of beats to process.")
def generate_commentary(
    snapshot_path: str,
    beats_path: str,
    backend: str,
    model_path: str | None,
    adapter_path: str | None,
    max_tokens: int,
    temperature: float,
    output: str | None,
    max_items: int | None,
):
    """Generate commentary for scheduled beats using a local model backend."""
    click.echo(f"Loading snapshots from {snapshot_path}...")
    snapshots = _load_snapshots_from_jsonl(snapshot_path, max_frames=None)

    click.echo(f"Loading beats from {beats_path}...")
    beats = _load_beats_from_jsonl(beats_path)

    selected = [b for b in beats if b.selected_for_generation]
    click.echo(f"Loaded {len(snapshots)} snapshots, {len(beats)} beats ({len(selected)} selected).")

    # Set up backend.
    gen_backend = load_backend(
        backend_type=backend,
        model_path=model_path,
        adapter_path=adapter_path,
    )
    click.echo(f"Backend: {gen_backend.model_variant}")

    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        backend=gen_backend.model_variant,
    )
    generator = CommentaryGenerator(gen_backend, config)

    # Generate.
    results = generator.generate_batch(snapshots, beats, max_items=max_items)

    click.echo(f"\n--- Generated {len(results)} commentaries ---\n")
    for r in results:
        click.echo(f"  Beat {r.beat_id}  [{r.event_type.value}]  {r.latency_ms:.1f}ms")
        click.echo(f"    {r.commentary_text}")
        click.echo("")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")
        click.echo(f"Wrote {len(results)} commentaries to {output}")


@cli.command("run-generator-over-beats")
@click.option(
    "--snapshots",
    "snapshot_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a RaceStateSnapshot JSONL file.",
)
@click.option(
    "--beats",
    "beats_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a ScheduledBeat JSONL file.",
)
@click.option("--max-items", type=int, default=20, help="Max beats to process.")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for generated commentary JSONL.",
)
@click.option(
    "--backend",
    type=click.Choice(["mock", "mlx", "llama_cpp"], case_sensitive=False),
    default="mock",
    help="Generator backend to use.",
)
@click.option("--model-path", type=click.Path(), default=None, help="Path to base model.")
@click.option("--adapter-path", type=click.Path(), default=None, help="Path to SFT adapter.")
def run_generator_over_beats(
    snapshot_path: str,
    beats_path: str,
    max_items: int,
    output: str | None,
    backend: str,
    model_path: str | None,
    adapter_path: str | None,
):
    """Run the commentary generator over a batch of scheduled beats.

    This is the main batch-processing entrypoint for demo and evaluation.
    """
    click.echo(f"=== Commentary Generation Run ===")
    click.echo(f"  Snapshots:  {snapshot_path}")
    click.echo(f"  Beats:      {beats_path}")
    click.echo(f"  Max items:  {max_items}")

    snapshots = _load_snapshots_from_jsonl(snapshot_path, max_frames=None)
    beats = _load_beats_from_jsonl(beats_path)

    gen_backend = load_backend(
        backend_type=backend,
        model_path=model_path,
        adapter_path=adapter_path,
    )
    click.echo(f"  Backend:    {gen_backend.model_variant}\n")

    generator = CommentaryGenerator(gen_backend)
    results = generator.generate_batch(snapshots, beats, max_items=max_items)

    # Summary table.
    total_latency = sum(r.latency_ms for r in results)
    avg_latency = total_latency / len(results) if results else 0
    avg_length = sum(r.output_length for r in results) / len(results) if results else 0

    for i, r in enumerate(results, 1):
        click.echo(f"[{i}/{len(results)}] Beat {r.beat_id}  |  {r.event_type.value}")
        click.echo(f"  Commentary: {r.commentary_text}")
        click.echo(f"  Latency: {r.latency_ms:.1f}ms  |  Length: {r.output_length} chars")
        click.echo("")

    click.echo(f"--- Summary ---")
    click.echo(f"  Total generated:  {len(results)}")
    click.echo(f"  Avg latency:      {avg_latency:.1f}ms")
    click.echo(f"  Avg output len:   {avg_length:.0f} chars")
    click.echo(f"  Total latency:    {total_latency:.1f}ms")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")
        click.echo(f"\nWrote {len(results)} commentaries to {output}")


def _load_commentaries_from_jsonl(
    path: str, max_items: int | None = None,
) -> list[GeneratedCommentary]:
    """Read GeneratedCommentary objects from a JSONL file."""
    items: list[GeneratedCommentary] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if max_items is not None and i >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            items.append(GeneratedCommentary.model_validate_json(line))
    return items


@cli.command("run-beat-manager")
@click.option(
    "--commentaries",
    "commentary_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a GeneratedCommentary JSONL file.",
)
@click.option(
    "--beats",
    "beats_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a ScheduledBeat JSONL file.",
)
@click.option(
    "--snapshots",
    "snapshot_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a RaceStateSnapshot JSONL file (optional, enriches output).",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for FinalCommentary JSONL.",
)
@click.option("--refresh-interval", type=int, default=3, help="Min frames between re-narrating a storyline.")
@click.option("--include-suppressed", is_flag=True, default=False, help="Include suppressed commentary in output.")
@click.option("--no-colour", is_flag=True, default=False, help="Disable ANSI colours in terminal output.")
def run_beat_manager(
    commentary_path: str,
    beats_path: str,
    snapshot_path: str | None,
    output: str | None,
    refresh_interval: int,
    include_suppressed: bool,
    no_colour: bool,
):
    """Run the BeatManager to produce final coherent commentary output.

    Takes generated commentary and scheduled beats, applies suppression
    and progression logic, and emits FinalCommentary objects.
    """
    click.echo("=== Beat Manager ===")
    click.echo(f"  Commentaries:     {commentary_path}")
    click.echo(f"  Beats:            {beats_path}")
    click.echo(f"  Snapshots:        {snapshot_path or '(none)'}")
    click.echo(f"  Refresh interval: {refresh_interval}")
    click.echo(f"  Include suppressed: {include_suppressed}\n")

    commentaries = _load_commentaries_from_jsonl(commentary_path)
    beats = _load_beats_from_jsonl(beats_path)
    snapshots = _load_snapshots_from_jsonl(snapshot_path, max_frames=None) if snapshot_path else None

    manager = BeatManager(
        refresh_interval=refresh_interval,
        include_suppressed=include_suppressed,
    )
    finals = manager.process_batch(beats, commentaries, snapshots)

    # Terminal output.
    adapter = TerminalAdapter(
        colour=not no_colour,
        show_suppressed=include_suppressed,
    )
    adapter.emit_batch(finals)
    adapter.print_summary()

    # JSONL output.
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for f in finals:
                fh.write(f.model_dump_json() + "\n")
        click.echo(f"Wrote {len(finals)} final commentaries to {output}")


@cli.command("stream-final-commentary")
@click.option(
    "--commentaries",
    "commentary_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a GeneratedCommentary JSONL file.",
)
@click.option(
    "--beats",
    "beats_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a ScheduledBeat JSONL file.",
)
@click.option(
    "--snapshots",
    "snapshot_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a RaceStateSnapshot JSONL file (optional).",
)
@click.option("--refresh-interval", type=int, default=3, help="Min frames between re-narrating a storyline.")
@click.option("--include-suppressed", is_flag=True, default=False, help="Include suppressed events in SSE stream.")
def stream_final_commentary(
    commentary_path: str,
    beats_path: str,
    snapshot_path: str | None,
    refresh_interval: int,
    include_suppressed: bool,
):
    """Stream final commentary as Server-Sent Events (SSE) to stdout.

    Output is in standard SSE wire format and can be piped to an HTTP
    response or consumed by an EventSource client.
    """
    commentaries = _load_commentaries_from_jsonl(commentary_path)
    beats = _load_beats_from_jsonl(beats_path)
    snapshots = _load_snapshots_from_jsonl(snapshot_path, max_frames=None) if snapshot_path else None

    manager = BeatManager(
        refresh_interval=refresh_interval,
        include_suppressed=include_suppressed,
    )
    finals = manager.process_batch(beats, commentaries, snapshots)

    sse = SSEAdapter(include_suppressed=include_suppressed)
    emitted = 0
    for f in finals:
        if sse.emit(f):
            emitted += 1

    # Final summary event.
    sys.stdout.write(f"event: done\ndata: {{\"total_emitted\": {emitted}}}\n\n")
    sys.stdout.flush()


if __name__ == "__main__":
    cli()
