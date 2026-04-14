"""Tests for the paper artifact generation module (Phase 16)."""

import json
import os
from pathlib import Path

import pytest

from f1_commentary.artifacts.exporter import (
    ArtifactExporter,
    generate_reproducibility_script,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

def _make_comparison() -> dict:
    """Return a synthetic comparison dict matching SystemComparison.compare() output."""
    return {
        "n_system_traces": 20,
        "n_baseline_traces": 20,
        "latency": {
            "system_mean_ms": 120.5,
            "baseline_mean_ms": 185.3,
            "system_p95_ms": 210.0,
            "baseline_p95_ms": 320.0,
        },
        "hallucination": {
            "system_rate": 0.08,
            "baseline_rate": 0.25,
        },
        "bert_score": {
            "system_f1": 0.847,
            "baseline_f1": 0.721,
            "delta_f1": 0.126,
        },
    }


def _make_examples() -> list[dict]:
    """Return synthetic qualitative examples."""
    return [
        {
            "label": "Overtake at Turn 1",
            "context": "Lap 15: VER within DRS range of HAM, gap 0.4s",
            "system_output": "Verstappen closes in on Hamilton through the DRS zone, the gap now just four tenths.",
            "baseline_output": "Verstappen is close to Hamilton.",
        },
        {
            "label": "Safety Car Restart",
            "context": "Lap 32: Safety car ending, field bunched up",
            "system_output": "The safety car peels in and the field roars back to racing speed — Leclerc will be watching his mirrors.",
            "baseline_output": "The safety car is coming in and racing resumes.",
        },
    ]


# ---------------------------------------------------------------------------
# 1. ArtifactExporter creates output directories
# ---------------------------------------------------------------------------

def test_exporter_creates_directories(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    assert exporter.figures_dir.is_dir()
    assert exporter.tables_dir.is_dir()
    assert exporter.examples_dir.is_dir()


# ---------------------------------------------------------------------------
# 2. generate_latex_metrics_table produces valid .tex file
# ---------------------------------------------------------------------------

def test_generate_latex_metrics_table(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    comparison = _make_comparison()
    path = exporter.generate_latex_metrics_table(comparison)

    assert path.exists()
    assert path.suffix == ".tex"

    content = path.read_text()
    assert r"\begin{table}" in content
    assert r"\end{table}" in content
    assert r"\begin{tabular}" in content
    assert "Our System" in content
    assert "Baseline" in content
    assert "Latency" in content
    assert "Hallucination" in content
    assert "BERTScore" in content
    # Check values are present
    assert "120.5" in content
    assert "185.3" in content
    assert "0.847" in content


# ---------------------------------------------------------------------------
# 3. generate_latency_chart produces .png file
# ---------------------------------------------------------------------------

def test_generate_latency_chart(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    sys_lat = [100.0, 120.0, 130.0, 110.0, 150.0, 200.0, 95.0, 105.0]
    base_lat = [180.0, 200.0, 220.0, 190.0, 250.0, 310.0, 170.0, 195.0]

    path = exporter.generate_latency_chart(sys_lat, base_lat)

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 4. generate_bert_score_chart produces .png file
# ---------------------------------------------------------------------------

def test_generate_bert_score_chart(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    sys_scores = [0.85, 0.88, 0.82, 0.90, 0.84]
    base_scores = [0.72, 0.70, 0.75, 0.68, 0.73]

    path = exporter.generate_bert_score_chart(sys_scores, base_scores)

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 5. generate_hallucination_chart produces .png file
# ---------------------------------------------------------------------------

def test_generate_hallucination_chart(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    path = exporter.generate_hallucination_chart(0.08, 0.25)

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. generate_qualitative_examples produces valid .tex file
# ---------------------------------------------------------------------------

def test_generate_qualitative_examples(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    examples = _make_examples()
    path = exporter.generate_qualitative_examples(examples)

    assert path.exists()
    assert path.suffix == ".tex"

    content = path.read_text()
    assert r"\begin{table}" in content
    assert r"\end{table}" in content
    assert "Overtake at Turn 1" in content
    assert "Safety Car Restart" in content
    assert "Verstappen" in content
    assert r"\textbf{Context}" in content
    assert r"\textbf{Our System}" in content
    assert r"\textbf{Baseline}" in content


# ---------------------------------------------------------------------------
# 7. generate_all returns dict of all paths
# ---------------------------------------------------------------------------

def test_generate_all(tmp_path):
    exporter = ArtifactExporter(tmp_path / "artifacts")
    comparison = _make_comparison()
    examples = _make_examples()

    sys_lat = [100.0, 120.0, 130.0, 110.0]
    base_lat = [180.0, 200.0, 220.0, 190.0]
    sys_bert = [0.85, 0.88, 0.82]
    base_bert = [0.72, 0.70, 0.75]

    paths = exporter.generate_all(
        comparison=comparison,
        system_latencies=sys_lat,
        baseline_latencies=base_lat,
        system_bert_scores=sys_bert,
        baseline_bert_scores=base_bert,
        examples=examples,
    )

    assert "metrics_table" in paths
    assert "latency_chart" in paths
    assert "bertscore_chart" in paths
    assert "hallucination_chart" in paths
    assert "qualitative_examples" in paths

    for name, path in paths.items():
        assert path.exists(), f"Artifact {name} not found at {path}"


# ---------------------------------------------------------------------------
# 8. generate_reproducibility_script creates executable .sh file
# ---------------------------------------------------------------------------

def test_generate_reproducibility_script(tmp_path):
    output_path = tmp_path / "scripts" / "reproduce.sh"
    path = generate_reproducibility_script(output_path)

    assert path.exists()
    assert path.suffix == ".sh"

    content = path.read_text()
    assert "#!/usr/bin/env bash" in content
    assert "set -euo pipefail" in content
    assert "evaluate-run" in content
    assert "compare-system-vs-baseline" in content
    assert "generate-artifacts" in content

    # Check executable permission
    assert os.access(path, os.X_OK)
