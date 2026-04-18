# Architecture

End-to-end map of the telemetry-to-narrative system. Phase numbers refer
to the project's build order; module paths are anchored at `code/src/`.

## Data flow

```
FastF1 session  ──┐
                   ├──► ReplayAdapter ──► TelemetryFrame stream
JSONL replay   ───┘                              │
                                                 ▼
                                          RaceStateEngine
                                          + FeatureExtractor
                                                 │
                                                 ▼
                                          RaceStateSnapshot
                                                 │
                                                 ▼
                                  snapshot_to_candidates.extract_candidates
                                                 │
                                                 ▼
                                          EditorialScheduler
                                                 │
                                  ┌──────────────┴──────────────┐
                                  ▼                             ▼
                       (LOCAL mode)                   (BASELINE mode)
                       CommentaryGenerator             BaselineGenerator
                       + local backend                 + provider backend
                                  │                             │
                                  ▼                             ▼
                            BeatManager                 (skipped — by design)
                                  │                             │
                                  ▼                             ▼
                      FinalCommentary               BaselineCommentary
                                  │                             │
                                  └──────────────┬──────────────┘
                                                 ▼
                              TerminalAdapter / SSEAdapter / JSONL traces
```

Every stage's output is a Pydantic record. IDs propagate end-to-end, so
any record in any trace can be joined back to its source frame.

## Modules

### Schemas — `telemetry_to_narrative/schemas/`
Pydantic v2 models that define the wire format between every stage.
The schemas are the contract; modules read and emit them, never
ad-hoc dicts.

- `telemetry_frame.py` — `TelemetryFrame`, `DriverState`, `GlobalState`,
  `SessionInfo`. One per replay tick.
- `race_state.py` — `RaceStateSnapshot` aggregating drivers, battles,
  flags, derived features for one frame.
- `candidate_event.py` — `CandidateEvent` plus the `EventType` enum
  (lead battle, overtake, pit, race control, …).
- `scheduled_beat.py` — `ScheduledBeat` produced by the scheduler,
  including selection / suppression and a score breakdown.
- `generated_commentary.py` — `GeneratedCommentary` from the local
  generator (carries prompt + output + timing).
- `baseline_commentary.py` — `BaselineCommentary` parallel to the local
  output but with provider/model fields.
- `final_commentary.py` — `FinalCommentary` after BeatManager applies
  storyline progression and suppression.
- `sft_dataset.py` — SFT example + dataset metadata records.

### Replay — `telemetry_to_narrative/adapters/`
- `session_loader.py` — wraps FastF1 caching/loading.
- `replay_adapter.py` — yields `TelemetryFrame`s at configurable
  speeds, optionally pacing at wall-clock.

### State — `telemetry_to_narrative/state/`
- `race_state_engine.py` — stateful engine that ingests one frame at a
  time and tracks per-driver position, battles, pit windows, flags.
- `feature_extractor.py` — converts engine state into a
  `RaceStateSnapshot` per frame.
- `inspection.py` — pretty-printing / CSV dumping helpers.

### Scheduler — `telemetry_to_narrative/scheduler/`
- `snapshot_to_candidates.py` — diff a snapshot against the previous
  one and emit `CandidateEvent`s.
- `editorial_scheduler.py` — score, prioritise, and suppress
  candidates; emits `ScheduledBeat`s with selection decisions.
- `config/scheduler_config.py` — narration thresholds, weights.

### Generation (local) — `telemetry_to_narrative/generation/`
- `model_loader.py` — `load_backend(backend_type, model_path,
  adapter_path)`. Falls back to `mock` on unknown types.
- `commentary_generator.py` — builds prompts for selected beats and
  invokes the backend, returning `GeneratedCommentary`.
- `prompt_builder.py` — per-event-type prompt templates.

### Baseline — `telemetry_to_narrative/baseline/`
- `backends.py` — `load_baseline_backend` (mock / openai-compatible).
- `baseline_generator.py` — frontier LLM path, parallel to the local
  generator. Outputs `BaselineCommentary`.
- `prompt_builder.py` — baseline prompts (deliberately minimal — the
  whole point of the baseline is "no structured BeatManager").

### Beat manager — `telemetry_to_narrative/beat_manager/`
- `beat_manager.py` — applies `refresh_interval`, dedupes per-storyline
  re-narration, and emits `FinalCommentary`. Optionally surfaces
  suppressed beats so traces stay inspectable.
- `output_adapters.py` — `TerminalAdapter` (ANSI) and `SSEAdapter`
  (Server-Sent Events).

### Pipeline — `telemetry_to_narrative/pipeline/`
- `config.py` — `PipelineConfig` dataclass + `PipelineMode`,
  `StopAfter`, `Scenario` enums. `validate()` fails fast on
  out-of-range numeric values.
- `stage_result.py` — `PipelineCounters` and `PipelineResult`.
- `trace_writer.py` — context-managed JSONL writer; one file per
  stage. No-op when `trace_dir` is `None`.
- `pipeline_runner.py` — `PipelineRunner` wires the stages together.
  Pure orchestration: no new logic.
- `diagnostics.py` — `run_doctor`, `validate_trace_linkage`,
  `check_trace_schemas`, `check_jsonl_schema`. All read-only.

### Training — `telemetry_to_narrative/training/`
- `dataset_builder.py` — joins snapshots + beats + transcripts into
  SFT examples.
- `transcript_loader.py` — multi-format transcript ingestion (JSON,
  JSONL, CSV, TXT).
- Alignment confidence: `exact`, `near`, `heuristic`, `unmatched`.

### CLI — `src/main.py`
Click group with commands grouped by purpose:

- Replay/state: `replay`, `build-state`, `inspect-state`
- Scheduler: `schedule-demo`, `schedule-inspect`
- Datasets: `build-sft-dataset`, `inspect-sft-dataset`, `dataset-stats`
- Local generation: `generate-commentary`, `run-generator-over-beats`
- Beat manager / streaming: `run-beat-manager`,
  `stream-final-commentary`
- Baseline: `run-baseline`, `compare-single-input`
- End-to-end pipeline: `run-pipeline`, `run-baseline-pipeline`,
  `run-scenario`
- Diagnostics: `doctor`, `validate-trace`

## Invariants

- **ID preservation.** `frame_id → snapshot.frame_id →
  candidate.source_frame_id → beat.source_frame_id → beat.beat_id →
  generation.beat_id + generation.generation_id → final.beat_id +
  final.source_generation_id`. The baseline path adds
  `baseline.beat_id` / `baseline.baseline_id`. `validate-trace`
  enforces this.
- **Suppressed-trace separation.** BeatManager inside `PipelineRunner`
  always runs with `include_suppressed=True`, so suppressed items hit
  the JSONL trace. Output-adapter visibility is gated separately by
  `include_suppressed_in_output`.
- **Graceful backend degradation.** `load_backend` and
  `load_baseline_backend` fall back to `mock` on unknown types, so
  typos never abort a run. `PipelineConfig.validate()` therefore does
  not check backend names — only numeric ranges.
- **No FastF1 in tests.** `PipelineRunner.run` accepts either a
  pre-built `frames` iterable or a `(year, gp, session)` triplet; the
  triplet path is the only branch that imports FastF1.
- **Schema-only contracts.** Every cross-module record is a Pydantic
  model. Adding a field is the only way two stages communicate new
  information.
