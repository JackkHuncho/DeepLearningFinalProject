# Curated demo examples

Three scenarios — **lead battle**, **pit strategy**, **race
control / safety car** — each captured end-to-end with prompt, local
output, baseline output, final output, and the trace IDs needed to
re-derive the example from the JSONL artifacts.

All three are reproducible from the bundled
`code/data/logs/demo_replay.jsonl` (a 20-frame sample with synthetic
pit stops added). The full trace dirs live under
`code/data/artifacts/demo_<scenario>_{local,baseline}/`.

To regenerate any example end-to-end:

```bash
cd code
python3 -m src.main run-pipeline \
  --from-jsonl data/logs/demo_replay.jsonl \
  --max-frames 20 --scenario <scenario> \
  --trace-dir data/artifacts/demo_<scenario>_local --no-terminal
python3 -m src.main run-baseline-pipeline \
  --from-jsonl data/logs/demo_replay.jsonl \
  --max-frames 20 --scenario <scenario> \
  --trace-dir data/artifacts/demo_<scenario>_baseline --no-terminal
```

> ID note: `beat_id`, `generation_id`, `final_id`, and `baseline_id`
> are UUIDs assigned per process. They link records *within* a single
> trace dir but differ across local↔baseline runs. Cross-run pairing
> uses the stable `(source_frame_id, event_type, involved_drivers)`
> tuple, plus the per-event `storyline_id` (e.g. `battle_PER_VER`,
> `pit_LEC`, `race_control`).

---

## 1. Lead battle — `battle_PER_VER`

| Field | Value |
|---|---|
| Scenario | `lead_battle` |
| Source frame | `0` |
| Storyline | `battle_PER_VER` |
| Drivers | VER, PER |
| Local `beat_id` | `478a8f6f571c` |
| Local `generation_id` | `6de11aa1b21b44d2` |
| Local `final_id` | `9daaa2fadb4d43e3` |
| Baseline `beat_id` | `3fe4e8b09ab3` |
| Baseline `baseline_id` | `bl_e993b745f8414e` |
| Local trace dir | `code/data/artifacts/demo_lead_battle_local/` |
| Baseline trace dir | `code/data/artifacts/demo_lead_battle_baseline/` |

### Local prompt

```
[Event: lead_battle | Drivers: VER, PER]

## Observed Facts
- Current lap: 1
- Race leader: VER
- Track status: GREEN FLAG
- VER: P1, on MEDIUM (lap 1), speed 306 kph
- PER: P2, on HARD (lap 1), gap ahead 0.6s, speed 307 kph
- Battle: PER chasing VER (gap 0.60s)
- Battle: HAM chasing PER (gap 0.90s)

## Derived Inferences
- (no inferences available)

## Context
- (none available)
## Storyline
- battle_PER_VER (ongoing)

## Task
Generate one short, professional-style commentary line grounded in the
provided race state. Keep it to one or two sentences.
```

### Local output (mock backend)
> Intense racing at the front between VER and PER, the gap barely
> changing lap after lap.

### Baseline output (mock provider)
> At the front of the field, VER continues to pressure PER, with the
> gap hovering just under one second.

### Final output (post-BeatManager)
> Intense racing at the front between VER and PER, the gap barely
> changing lap after lap.
> *(suppression_applied=False, progression=initial_interest)*

---

## 2. Pit strategy — `pit_LEC`

| Field | Value |
|---|---|
| Scenario | `pit_strategy` |
| Source frame | `14` |
| Storyline | `pit_LEC` |
| Drivers | LEC |
| Local `beat_id` | `080a55842ba1` |
| Local `generation_id` | `85239059ee75423c` |
| Local `final_id` | `f14241bd810f483a` |
| Baseline `beat_id` | `7ddd9619bce3` |
| Baseline `baseline_id` | `bl_9059e51a9b164f` |
| Local trace dir | `code/data/artifacts/demo_pit_strategy_local/` |
| Baseline trace dir | `code/data/artifacts/demo_pit_strategy_baseline/` |

### Local prompt

```
[Event: pit_strategy | Drivers: LEC]

## Observed Facts
- Current lap: 15
- Race leader: VER
- Track status: GREEN FLAG
- LEC: P5, on HARD (lap 0), gap ahead 1.7s, speed 313 kph
- Battle: LEC chasing RUS (gap 1.70s)
- Battle: SAI chasing LEC (gap 2.00s)

## Derived Inferences
- LEC pace is improving (trend -0.107)
- Tire compound change — strategic pit stop

## Context
- (none available)
## Storyline
- pit_LEC (ongoing)

## Task
Generate one short, professional-style commentary line grounded in the
provided race state. Keep it to one or two sentences.
```

### Local output (mock backend)
> LEC pits for fresh rubber. The strategy call could be decisive.

### Baseline output (mock provider)
> LEC has committed to a pit stop, rolling the dice on strategy as the
> window opens.

### Final output
> LEC pits for fresh rubber. The strategy call could be decisive.
> *(suppression_applied=False, progression=initial_interest)*

> The "Tire compound change — strategic pit stop" inference is
> structured-pipeline content the baseline doesn't get for free.

---

## 3. Race control / safety car — `race_control`

| Field | Value |
|---|---|
| Scenario | `race_control` |
| Source frame | `0` |
| Storyline | `race_control` |
| Drivers | (field) |
| Local `beat_id` | `3790caf1eaf4` |
| Local `generation_id` | `e4680f08301f46f8` |
| Local `final_id` | `cb4c5eda42ce44f2` |
| Baseline `beat_id` | `b2b60b138642` |
| Baseline `baseline_id` | `bl_b186054a23ed46` |
| Local trace dir | `code/data/artifacts/demo_race_control_local/` |
| Baseline trace dir | `code/data/artifacts/demo_race_control_baseline/` |

### Local prompt

```
[Event: race_control | Drivers: field]

## Observed Facts
- Current lap: 1
- Race leader: VER
- Track status: GREEN FLAG

## Derived Inferences
- Race control intervention — track conditions changed

## Context
- (none available)
## Storyline
- race_control (ongoing)

## Task
Generate one short, professional-style commentary line grounded in the
provided race state. Keep it to one or two sentences.
```

### Local output (mock backend)
> Race control intervenes — the safety car is deployed.

### Baseline output (mock provider)
> Race control has intervened — the safety car has been deployed and
> the field will be neutralised.

### Final output
> Race control intervenes — the safety car is deployed.
> *(suppression_applied=False, progression=initial_interest)*

---

## How to follow the IDs

Within a single trace dir, the chain is:

```
TelemetryFrame.frame_id
  → RaceStateSnapshot.frame_id
  → CandidateEvent.source_frame_id
  → ScheduledBeat.source_frame_id
  → ScheduledBeat.beat_id
  → GeneratedCommentary.beat_id (+ generation_id)
  → FinalCommentary.beat_id (+ source_generation_id, final_id)
```

Baseline mode replaces the last two steps with
`BaselineCommentary.beat_id` (+ `baseline_id`).
`validate-trace --trace-dir <dir>` enforces the chain.

To pull a single example by hand:

```bash
TRACE=code/data/artifacts/demo_pit_strategy_local
grep '"beat_id":"080a55842ba1"' $TRACE/scheduled_beats.jsonl
grep '"generation_id":"85239059ee75423c"' $TRACE/generated_commentary.jsonl
grep '"final_id":"f14241bd810f483a"' $TRACE/final_commentary.jsonl
```
