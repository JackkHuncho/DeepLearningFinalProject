#!/usr/bin/env bash
set -euo pipefail

echo "=== Reproducing paper artifacts ==="

# Evaluate system traces
python -m f1_commentary.cli evaluate-run \
    --traces data/logs/traces/system_traces.jsonl \
    --output data/artifacts/system_results.json

# Evaluate baseline traces
python -m f1_commentary.cli evaluate-run \
    --traces data/logs/traces/baseline_traces.jsonl \
    --output data/artifacts/baseline_results.json

# Compare
python -m f1_commentary.cli compare-system-vs-baseline \
    --system-traces data/logs/traces/system_traces.jsonl \
    --baseline-traces data/logs/traces/baseline_traces.jsonl \
    --output data/artifacts/comparison.json

# Generate artifacts
python -m f1_commentary.cli generate-artifacts \
    --comparison data/artifacts/comparison.json \
    --output data/artifacts/paper

echo "=== Done. Artifacts in data/artifacts/paper/ ==="
