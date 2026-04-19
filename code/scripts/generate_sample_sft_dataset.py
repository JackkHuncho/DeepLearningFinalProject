#!/usr/bin/env python3
"""Generate sample SFT dataset files from existing sample data.

Uses the sample snapshots, runs the scheduler to get beats, then builds
the SFT dataset with the sample transcript file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the code directory is on the path.
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from src.telemetry_to_narrative.schemas.race_state import RaceStateSnapshot
from src.telemetry_to_narrative.scheduler.editorial_scheduler import EditorialScheduler
from src.telemetry_to_narrative.scheduler.snapshot_to_candidates import extract_candidates
from src.telemetry_to_narrative.training.dataset_builder import build_dataset


def main():
    snapshot_path = CODE_DIR / "data" / "logs" / "sample_snapshots.jsonl"
    beats_path = CODE_DIR / "data" / "logs" / "sample_scheduled_beats.jsonl"
    transcript_path = CODE_DIR / "data" / "transcripts" / "sample_commentary.json"
    output_dir = CODE_DIR / "data" / "datasets"

    if not snapshot_path.exists():
        print(f"ERROR: {snapshot_path} not found. Run generate_sample_jsonl.py first.")
        sys.exit(1)
    if not beats_path.exists():
        print(f"ERROR: {beats_path} not found. Run schedule-demo first.")
        sys.exit(1)
    if not transcript_path.exists():
        print(f"ERROR: {transcript_path} not found.")
        sys.exit(1)

    print(f"Building SFT dataset...")
    print(f"  Snapshots:   {snapshot_path}")
    print(f"  Beats:       {beats_path}")
    print(f"  Transcript:  {transcript_path}")
    print(f"  Output:      {output_dir}")

    metadata = build_dataset(
        snapshot_path=snapshot_path,
        beats_path=beats_path,
        transcript_paths=[transcript_path],
        output_dir=output_dir,
        alignment_mode="loose",
        tolerance_sec=30.0,  # generous for sample data
    )

    print(f"\nDataset Summary:")
    print(f"  Total examples:  {metadata.total_examples}")
    print(f"  Train:           {metadata.train_examples}")
    print(f"  Val:             {metadata.val_examples}")
    print(f"  Event types:     {dict(metadata.event_type_distribution)}")
    print(f"  Alignment:       {dict(metadata.alignment_distribution)}")
    print(f"  Quality:         {metadata.quality_report}")
    print(f"\nFiles written to {output_dir}/")


if __name__ == "__main__":
    main()
