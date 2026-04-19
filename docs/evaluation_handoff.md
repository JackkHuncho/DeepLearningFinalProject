# Evaluation handoff

What the even-phase evaluation code can rely on from the odd-phase
pipeline. This is the contract — if any of it changes, this doc moves
with the change.

## Artifact folder layout

Every pipeline run writes one self-contained directory:

```
<trace-dir>/
  replay_frames.jsonl          # TelemetryFrame
  snapshots.jsonl              # RaceStateSnapshot
  candidate_events.jsonl       # CandidateEvent
  scheduled_beats.jsonl        # ScheduledBeat (selected + suppressed)
  generated_commentary.jsonl   # GeneratedCommentary  (local mode only)
  final_commentary.jsonl       # FinalCommentary      (local mode only)
  baseline_commentary.jsonl    # BaselineCommentary   (baseline mode only)
  run_summary.json             # mode, counters, flags, file paths
```

- All `*.jsonl` files are line-delimited JSON.
- Empty files are still created, so loaders can `open()` unconditionally.
- `run_summary.json["mode"]` is `"local"` or `"baseline"` —
  use this to decide whether to read `final_commentary.jsonl` or
  `baseline_commentary.jsonl`.
- The *traces are schema-aligned across modes* — you do not need a
  separate ingest path for baseline.

Reference dirs (committed):

| Dir | Mode | Scenario |
|---|---|---|
| `code/data/artifacts/demo_lead_battle_local/` | local | lead_battle |
| `code/data/artifacts/demo_lead_battle_baseline/` | baseline | lead_battle |
| `code/data/artifacts/demo_pit_strategy_local/` | local | pit_strategy |
| `code/data/artifacts/demo_pit_strategy_baseline/` | baseline | pit_strategy |
| `code/data/artifacts/demo_race_control_local/` | local | race_control |
| `code/data/artifacts/demo_race_control_baseline/` | baseline | race_control |
| `code/data/artifacts/sample_run_local/` | local | all |
| `code/data/artifacts/sample_run_baseline/` | baseline | all |

## Shared schema keys

Every record carries enough to be joined back to its source. Stable
across runs:

| Key | Lives on | Stable across local↔baseline? |
|---|---|---|
| `frame_id` (int) | TelemetryFrame, RaceStateSnapshot | ✅ |
| `source_frame_id` (int) | CandidateEvent, ScheduledBeat | ✅ |
| `event_type` (enum str) | CandidateEvent, ScheduledBeat, GeneratedCommentary, FinalCommentary, BaselineCommentary | ✅ |
| `involved_drivers` (list[str]) | same as above | ✅ |
| `storyline_id` (str) | ScheduledBeat, GeneratedCommentary, FinalCommentary, BaselineCommentary | ✅ |
| `beat_id` (UUID) | ScheduledBeat, GeneratedCommentary, FinalCommentary, BaselineCommentary | ❌ — process-local |
| `generation_id` (UUID) | GeneratedCommentary, FinalCommentary.source_generation_id | local-mode only |
| `final_id` (UUID) | FinalCommentary | local-mode only |
| `baseline_id` (UUID) | BaselineCommentary | baseline-mode only |

**To pair a local record with its baseline counterpart**, match on
`(source_frame_id, event_type, sorted(involved_drivers))` — or, for
ongoing storylines, `storyline_id`. **Never join on `beat_id` across
processes.**

## Within-trace linkage (single dir)

```
frame_id
  ↓
RaceStateSnapshot.frame_id
  ↓
CandidateEvent.source_frame_id
  ↓
ScheduledBeat.source_frame_id        (and beat.beat_id is fresh here)
  ↓
GeneratedCommentary.beat_id          (gets its own generation_id)
  ↓
FinalCommentary.beat_id  +  FinalCommentary.source_generation_id
                         (gets its own final_id)
```

Baseline mode replaces the last two rows with
`BaselineCommentary.beat_id` (+ `baseline_id`).

This is exactly what `validate-trace` checks. Run it on any new
artifact dir before evaluating:

```bash
python3 -m src.main validate-trace --trace-dir <dir>
python3 -m src.main validate-trace --trace-dir <dir> --json   # for CI
```

Exit code is `0` on success, non-zero on any linkage or schema error.

## Reference loader

The pipeline's diagnostics module exposes the loaders the validators
themselves use. Evaluation code should re-use them instead of writing
its own:

```python
from src.telemetry_to_narrative.pipeline.diagnostics import (
    check_jsonl_schema,
    check_trace_schemas,
    load_run_summary,
    validate_trace_linkage,
    TRACE_SCHEMAS,        # filename → Pydantic schema mapping
)
from src.telemetry_to_narrative.schemas.generated_commentary import GeneratedCommentary
from src.telemetry_to_narrative.schemas.final_commentary import FinalCommentary
from src.telemetry_to_narrative.schemas.baseline_commentary import BaselineCommentary

# Load every record of a stage as Pydantic objects.
def load_jsonl(path, schema_cls):
    with open(path) as fh:
        return [
            schema_cls.model_validate_json(line)
            for line in fh if line.strip()
        ]
```

`TRACE_SCHEMAS` (in `pipeline/diagnostics.py`) is the authoritative
filename → schema map. If you need to add a new stage to evaluation,
add it there first.

## Common evaluation joins

Pairing one local generation with the matching baseline output:

```python
local = load_jsonl("…/demo_pit_strategy_local/generated_commentary.jsonl", GeneratedCommentary)
base  = load_jsonl("…/demo_pit_strategy_baseline/baseline_commentary.jsonl", BaselineCommentary)

key = lambda r: (r.event_type, tuple(sorted(r.involved_drivers)), r.storyline_id)
local_by_key = {key(r): r for r in local}
pairs = [(local_by_key[key(b)], b) for b in base if key(b) in local_by_key]
```

Pairing one final commentary with the upstream beat + snapshot:

```python
finals = load_jsonl("…/final_commentary.jsonl", FinalCommentary)
beats  = {b.beat_id: b for b in load_jsonl("…/scheduled_beats.jsonl", ScheduledBeat)}
snaps  = {s.frame_id: s for s in load_jsonl("…/snapshots.jsonl", RaceStateSnapshot)}

for f in finals:
    beat = beats[f.beat_id]
    snap = snaps[beat.source_frame_id]
    # … score f.final_text against snap (faithfulness) or beat
    #   (relevance to scheduling decision)
```

## What's known to be stable vs. in-flux

**Stable** (not changing in the project's remaining phases):

- The eight `<trace-dir>/*.jsonl` filenames + `run_summary.json`.
- All Pydantic schema fields listed above.
- The `validate-trace` exit-code contract.
- The scenario enum: `all | lead_battle | pit_strategy | race_control`.
- The `mode` enum: `local | baseline`.

**Likely to grow but backwards-compatible**:

- New optional fields on existing schemas (Pydantic ignores unknown
  fields if `model_config = ConfigDict(extra="ignore")`).
- New entries in `run_summary.json["counters"]`.

**Will change** (don't depend on these for evaluation):

- The mock backends' commentary text — placeholder until the SFT
  adapter is wired.
- Specific `beat_id` / `generation_id` values — UUIDs.
- The contents of `data/logs/demo_replay.jsonl` may grow as we add
  scenarios.

## Sanity-checking before each evaluation run

```bash
cd code
python3 -m src.main doctor                              # env health
python3 -m src.main validate-trace --trace-dir <dir>    # this run
```

If `doctor` reports `Overall: OK` and `validate-trace` reports
`Overall: OK`, the artifact set is safe to feed into the evaluation
pipeline.
