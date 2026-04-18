# Telemetry-to-Narrative

Converts F1 session telemetry into live-style commentary through a
structured pipeline: replay → race-state → scheduler → generator →
beat manager → final output. A frontier-LLM baseline path supports
A/B comparison.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a module-by-module map and
the system invariants enforced across stages.

## Install

```bash
cd code
pip install -r requirements.txt   # if present, else use venv of choice
```

Run the test suite:

```bash
cd code && python3 -m pytest -q
```

Sanity-check the local environment (folders, optional deps, backends,
sample artifacts):

```bash
cd code && python3 -m src.main doctor
```

## Quickstart

The end-to-end pipeline runs through `PipelineRunner`, exposed as three
CLI wrappers:

```bash
# 1. Full local pipeline — replay → final commentary
python3 -m src.main run-pipeline \
  --year 2024 --gp Monza --session R \
  --max-frames 50 --max-items 20 \
  --trace-dir data/artifacts/run_$(date +%Y%m%d_%H%M%S)

# 2. Frontier-baseline pipeline — BeatManager is skipped by design
python3 -m src.main run-baseline-pipeline \
  --year 2024 --gp Monza --session R \
  --max-frames 50 --max-items 20 \
  --baseline-backend openai --baseline-model gpt-4o-mini \
  --trace-dir data/artifacts/baseline_$(date +%Y%m%d_%H%M%S)

# 3. Scenario-focused demo — filter candidates to one event type
python3 -m src.main run-scenario --type lead_battle \
  --year 2024 --gp Monza --session R --max-frames 50
```

Run against a pre-captured replay JSONL instead of FastF1:

```bash
python3 -m src.main run-pipeline \
  --from-jsonl data/logs/sample_replay.jsonl \
  --max-frames 20 --trace-dir data/artifacts/my_run
```

## Sample command sequence

A reproducible end-to-end demo using only the bundled sample replay —
no FastF1, no API keys, no GPUs:

```bash
cd code

# 1. Health check.
python3 -m src.main doctor

# 2. Local pipeline against the bundled replay.
python3 -m src.main run-pipeline \
  --from-jsonl data/logs/sample_replay.jsonl \
  --max-frames 20 \
  --trace-dir data/artifacts/demo_local \
  --no-terminal

# 3. Baseline pipeline (mock provider — same shape, no API call).
python3 -m src.main run-baseline-pipeline \
  --from-jsonl data/logs/sample_replay.jsonl \
  --max-frames 20 \
  --trace-dir data/artifacts/demo_baseline \
  --no-terminal

# 4. Validate the traces — IDs link cleanly, schemas match.
python3 -m src.main validate-trace --trace-dir data/artifacts/demo_local
python3 -m src.main validate-trace --trace-dir data/artifacts/demo_baseline

# 5. Inspect a single (snapshot, beat) pair side-by-side.
python3 -m src.main compare-single-input \
  --snapshot data/artifacts/demo_local/snapshots.jsonl \
  --beat data/artifacts/demo_local/scheduled_beats.jsonl
```

### Useful flags

| Flag | Purpose |
|---|---|
| `--stop-after {replay,snapshots,beats,generation,final}` | Debug — stop after any stage |
| `--scenario {all,lead_battle,pit_strategy,race_control}` | Filter candidates |
| `--enable-retrieval` / `--no-retrieval` | Toggle retrieval (placeholder) |
| `--enable-grounding` / `--no-grounding` | Toggle grounding guard (placeholder) |
| `--enable-beat-manager` / `--no-beat-manager` | Toggle BeatManager |
| `--terminal` / `--no-terminal` | ANSI terminal adapter |
| `--sse` / `--no-sse` | Server-Sent Events adapter |
| `--local-backend {mock,mlx,llama_cpp}` | Local model backend |
| `--baseline-backend {mock,openai}` | Baseline provider |
| `--speed 1x\|5x\|10x --realtime` | Pace replay at wall-clock |
| `--include-suppressed` | Surface suppressed beats in output |

Every option has full `--help` text — e.g.
`python3 -m src.main run-pipeline --help`.

## Trace artifacts

Each run writes:

```
<trace-dir>/
  replay_frames.jsonl          # TelemetryFrame records
  snapshots.jsonl              # RaceStateSnapshot
  candidate_events.jsonl       # CandidateEvent
  scheduled_beats.jsonl        # ScheduledBeat (selected + suppressed)
  generated_commentary.jsonl   # GeneratedCommentary (local mode)
  final_commentary.jsonl       # FinalCommentary (emitted + suppressed)
  baseline_commentary.jsonl    # BaselineCommentary (baseline mode)
  run_summary.json             # mode, counters, flags, file paths
```

Sample reference artifacts:

- `code/data/artifacts/sample_run_local/`
- `code/data/artifacts/sample_run_baseline/`

IDs are preserved across every stage so any trace can be joined back
to its source (`frame_id → beat.source_frame_id → generation.beat_id →
final.source_generation_id`). Suppressed items are always traced even
when hidden from the terminal/SSE adapters.

`validate-trace` enforces these invariants:

```bash
python3 -m src.main validate-trace --trace-dir data/artifacts/sample_run_local
```

It loads each JSONL stage, validates every row against its Pydantic
schema, and verifies that all referenced IDs resolve.

## Training (SFT dataset)

The local generator is meant to be fine-tuned with supervised
examples. The dataset path is:

```bash
# Build an SFT dataset from snapshots + beats + transcripts.
python3 -m src.main build-sft-dataset \
  --snapshots data/artifacts/sample_run_local/snapshots.jsonl \
  --beats data/artifacts/sample_run_local/scheduled_beats.jsonl \
  --transcript data/transcripts/2024_monza_r.json \
  --output-dir data/datasets/monza_r \
  --alignment-mode loose --tolerance 10.0 \
  --min-confidence near

# Inspect or summarise the result.
python3 -m src.main inspect-sft-dataset \
  --from-jsonl data/datasets/monza_r/sft_train.jsonl
python3 -m src.main dataset-stats --dataset-dir data/datasets/monza_r
```

## Evaluation / comparison

Run the same `(snapshot, beat)` through both the local and baseline
paths and inspect prompts and outputs side-by-side:

```bash
python3 -m src.main compare-single-input \
  --snapshot data/artifacts/sample_run_local/snapshots.jsonl \
  --beat data/artifacts/sample_run_local/scheduled_beats.jsonl \
  --local-backend mock --baseline-backend mock
```

For a batch view, point the local and baseline pipelines at the same
input (`--from-jsonl`) and compare the resulting JSONL traces. The
trace files are schema-aligned across modes — no special-case
ingestion is needed.

## Diagnostics commands

| Command | Purpose |
|---|---|
| `doctor` | Folders, optional deps, backend reachability, sample artifacts |
| `validate-trace --trace-dir DIR` | ID linkage + JSONL schema check across a run |

Both commands accept `--json` for machine-readable output.
