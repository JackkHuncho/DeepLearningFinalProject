"""Dataset builder — orchestrates the full SFT dataset creation pipeline.

End-to-end flow
===============
1. Load RaceStateSnapshots from JSONL.
2. Load ScheduledBeats from JSONL.
3. Build EventWindows around selected beats.
4. Load and align transcript segments.
5. Format SFT examples.
6. Run quality checks and deduplication.
7. Split into train / val sets.
8. Write output JSONL and metadata JSON.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.schemas.scheduled_beat import ScheduledBeat
from src.telemetry_to_narrative.schemas.sft_dataset import (
    AlignmentConfidence,
    DatasetMetadata,
    SFTExample,
)
from src.telemetry_to_narrative.training.event_window_builder import extract_event_windows
from src.telemetry_to_narrative.training.transcript_loader import load_transcript
from src.telemetry_to_narrative.training.transcript_aligner import align_transcripts
from src.telemetry_to_narrative.training.sft_formatter import format_sft_dataset
from src.telemetry_to_narrative.training.quality_checks import run_quality_checks

logger = logging.getLogger(__name__)


def _load_snapshots(path: Path) -> list[RaceStateSnapshot]:
    snapshots = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                snapshots.append(RaceStateSnapshot.model_validate_json(line))
    return snapshots


def _load_beats(path: Path) -> list[ScheduledBeat]:
    beats = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                beats.append(ScheduledBeat.model_validate_json(line))
    return beats


def split_dataset(
    examples: list[SFTExample],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split examples into train and val sets.

    Uses a deterministic shuffle with the given seed for reproducibility.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_fraction)))
    return shuffled[:split_idx], shuffled[split_idx:]


def _write_jsonl(examples: list[SFTExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")


def build_dataset(
    snapshot_path: str | Path,
    beats_path: str | Path,
    transcript_paths: list[str | Path],
    output_dir: str | Path,
    context_radius: int = 2,
    alignment_mode: str = "loose",
    tolerance_sec: float = 10.0,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    min_target_words: int = 5,
    max_target_words: int = 500,
    min_confidence: AlignmentConfidence = AlignmentConfidence.NEAR,
    reference_date: Optional[datetime] = None,
) -> DatasetMetadata:
    """Run the full dataset creation pipeline.

    Parameters
    ----------
    snapshot_path : Path
        JSONL file of RaceStateSnapshots.
    beats_path : Path
        JSONL file of ScheduledBeats.
    transcript_paths : list[Path]
        Transcript files to load (auto-detected by extension).
    output_dir : Path
        Directory for output files (sft_train.jsonl, sft_val.jsonl, sft_metadata.json).
    context_radius : int
        Frames on each side of beat centre for EventWindow.
    alignment_mode : str
        "strict" or "loose" for transcript alignment.
    tolerance_sec : float
        Near-match tolerance in seconds.
    val_fraction : float
        Fraction of data for validation split.
    split_seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    DatasetMetadata
        Metadata about the generated dataset.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs.
    logger.info("Loading snapshots from %s", snapshot_path)
    snapshots = _load_snapshots(Path(snapshot_path))

    logger.info("Loading beats from %s", beats_path)
    beats = _load_beats(Path(beats_path))

    # 2. Build event windows.
    windows = extract_event_windows(snapshots, beats, context_radius=context_radius)
    logger.info("Built %d event windows.", len(windows))

    # 3. Load transcripts.
    all_segments = []
    transcript_source_names = []
    for tp in transcript_paths:
        tp = Path(tp)
        segments = load_transcript(tp, reference_date=reference_date)
        all_segments.extend(segments)
        transcript_source_names.append(tp.name)
    all_segments.sort(key=lambda s: s.timestamp)
    logger.info("Loaded %d transcript segments from %d files.", len(all_segments), len(transcript_paths))

    # 4. Align transcripts to windows.
    alignments = align_transcripts(
        windows, all_segments,
        mode=alignment_mode,
        tolerance_sec=tolerance_sec,
    )

    # 5. Format SFT examples.
    examples = format_sft_dataset(alignments)

    # 6. Quality checks.
    examples, quality_report = run_quality_checks(
        examples,
        min_target_words=min_target_words,
        max_target_words=max_target_words,
        min_confidence=min_confidence,
    )

    # 7. Split.
    train, val = split_dataset(examples, val_fraction=val_fraction, seed=split_seed)

    # 8. Write outputs.
    _write_jsonl(train, out / "sft_train.jsonl")
    _write_jsonl(val, out / "sft_val.jsonl")
    logger.info("Wrote %d train, %d val examples to %s", len(train), len(val), out)

    # Build metadata.
    event_dist: Counter[str] = Counter()
    align_dist: Counter[str] = Counter()
    sessions_seen: list[dict] = []
    session_keys: set[str] = set()
    for ex in examples:
        event_dist[ex.event_type.value] += 1
        align_dist[ex.alignment_confidence.value] += 1
        key = f"{ex.session_info.year}_{ex.session_info.grand_prix}_{ex.session_info.session_type}"
        if key not in session_keys:
            session_keys.add(key)
            sessions_seen.append({
                "year": ex.session_info.year,
                "grand_prix": ex.session_info.grand_prix,
                "session_type": ex.session_info.session_type,
            })

    metadata = DatasetMetadata(
        created_at=datetime.now(timezone.utc),
        total_examples=len(examples),
        train_examples=len(train),
        val_examples=len(val),
        sessions=sessions_seen,
        event_type_distribution=dict(event_dist),
        alignment_distribution=dict(align_dist),
        quality_report=quality_report.get("summary", {}),
        split_seed=split_seed,
        val_fraction=val_fraction,
        transcript_sources=transcript_source_names,
    )

    meta_path = out / "sft_metadata.json"
    with open(meta_path, "w") as fh:
        fh.write(metadata.model_dump_json(indent=2))
    logger.info("Wrote metadata to %s", meta_path)

    return metadata
