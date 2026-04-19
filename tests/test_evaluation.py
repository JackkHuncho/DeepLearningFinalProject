"""Tests for the evaluation module (Phase 14)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from f1_commentary.evaluation.metrics import (
    BERTScoreResult,
    compute_bert_score,
    compute_hallucination_rate,
    compute_latency_metrics,
    generate_human_eval_forms,
    aggregate_human_eval,
)
from f1_commentary.evaluation.evaluator import TraceEvaluator
from f1_commentary.evaluation.compare import SystemComparison


# ---------------------------------------------------------------------------
# Helpers — synthetic trace data
# ---------------------------------------------------------------------------

def _make_trace(
    total_latency_ms: float = 100.0,
    final_text: str = "Verstappen overtakes Hamilton into turn 1.",
    claims: list[dict] | None = None,
) -> dict:
    """Return a minimal trace dict matching TraceEntry-like structure."""
    return {
        "trace_id": "t-001",
        "beat_id": "b-001",
        "total_latency_ms": total_latency_ms,
        "final_commentary": {"final_text": final_text},
        "guard_decision": {
            "claims": claims or [],
        },
    }


def _write_traces(path: Path, traces: list[dict]) -> None:
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------------------
# 1. compute_latency_metrics — correct mean/median/p95/p99
# ---------------------------------------------------------------------------

def test_latency_metrics_correct_values():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    result = compute_latency_metrics(latencies)
    assert result["mean"] == 55.0
    assert result["median"] == 55.0
    assert result["max"] == 100.0
    # p95 index = int(10 * 0.95) = 9 -> sorted[9] = 100
    assert result["p95"] == 100.0
    # p99 index = int(10 * 0.99) = 9 -> sorted[9] = 100
    assert result["p99"] == 100.0


# ---------------------------------------------------------------------------
# 2. compute_latency_metrics — empty list returns zeros
# ---------------------------------------------------------------------------

def test_latency_metrics_empty():
    result = compute_latency_metrics([])
    assert result == {"mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}


# ---------------------------------------------------------------------------
# 3. compute_hallucination_rate — counts unsupported claims correctly
# ---------------------------------------------------------------------------

def test_hallucination_rate_counts():
    traces = [
        _make_trace(claims=[
            {"claim": "VER overtook HAM", "support_type": "observed"},
            {"claim": "VER set fastest lap", "support_type": "unsupported"},
        ]),
        _make_trace(claims=[
            {"claim": "HAM pitted", "support_type": "retrieved"},
            {"claim": "HAM on softs", "support_type": "unsupported"},
            {"claim": "HAM fastest sector", "support_type": "unsupported"},
        ]),
    ]
    result = compute_hallucination_rate(traces)
    assert result["total_claims"] == 5
    assert result["unsupported_claims"] == 3
    assert result["hallucination_rate"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 4. compute_hallucination_rate — no claims returns 0
# ---------------------------------------------------------------------------

def test_hallucination_rate_no_claims():
    traces = [_make_trace(claims=[]), _make_trace(claims=[])]
    result = compute_hallucination_rate(traces)
    assert result["total_claims"] == 0
    assert result["unsupported_claims"] == 0
    assert result["hallucination_rate"] == 0.0


# ---------------------------------------------------------------------------
# 5. generate_human_eval_forms — creates correct number and writes JSON
# ---------------------------------------------------------------------------

def test_generate_human_eval_forms(tmp_path):
    sys_outputs = [
        {"text": "Great overtake by VER", "context": "Lap 15"},
        {"text": "HAM pits for hards", "context": "Lap 22"},
    ]
    base_outputs = [
        {"text": "VER passes HAM"},
        {"text": "HAM makes a pit stop"},
    ]
    out = tmp_path / "forms.json"
    forms = generate_human_eval_forms(sys_outputs, base_outputs, out)

    assert len(forms) == 2
    assert forms[0].example_id == "eval_000"
    assert forms[1].system_output == "HAM pits for hards"
    assert out.exists()

    with open(out) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["race_context"] == "Lap 15"


# ---------------------------------------------------------------------------
# 6. aggregate_human_eval — computes averages correctly
# ---------------------------------------------------------------------------

def test_aggregate_human_eval(tmp_path):
    forms = [
        {
            "example_id": "eval_000",
            "system_output": "a",
            "baseline_output": "b",
            "race_context": "c",
            "fluency_system": 5,
            "fluency_baseline": 3,
            "accuracy_system": 4,
            "accuracy_baseline": 2,
            "insight_system": 4,
            "insight_baseline": 3,
            "preference": "system",
        },
        {
            "example_id": "eval_001",
            "system_output": "d",
            "baseline_output": "e",
            "race_context": "f",
            "fluency_system": 3,
            "fluency_baseline": 4,
            "accuracy_system": 2,
            "accuracy_baseline": 3,
            "insight_system": 2,
            "insight_baseline": 4,
            "preference": "baseline",
        },
    ]
    forms_path = tmp_path / "completed.json"
    with open(forms_path, "w") as f:
        json.dump(forms, f)

    result = aggregate_human_eval(forms_path)
    assert result["n_evaluated"] == 2
    assert result["avg_fluency_system"] == 4.0
    assert result["avg_fluency_baseline"] == 3.5
    assert result["avg_accuracy_system"] == 3.0
    assert result["preference_system"] == 1
    assert result["preference_baseline"] == 1


# ---------------------------------------------------------------------------
# 7. TraceEvaluator — loads JSONL traces and runs evaluate()
# ---------------------------------------------------------------------------

def test_trace_evaluator_evaluate(tmp_path):
    traces = [
        _make_trace(total_latency_ms=100.0, claims=[
            {"claim": "x", "support_type": "observed"},
        ]),
        _make_trace(total_latency_ms=200.0, claims=[
            {"claim": "y", "support_type": "unsupported"},
        ]),
    ]
    trace_file = tmp_path / "traces.jsonl"
    _write_traces(trace_file, traces)

    evaluator = TraceEvaluator(trace_file)
    results = evaluator.evaluate()

    assert results["n_traces"] == 2
    assert results["latency"]["mean"] == 150.0
    assert results["hallucination"]["total_claims"] == 2
    assert results["hallucination"]["unsupported_claims"] == 1
    assert "bert_score" not in results  # no references provided


# ---------------------------------------------------------------------------
# 8. TraceEvaluator — export_results writes JSON
# ---------------------------------------------------------------------------

def test_trace_evaluator_export(tmp_path):
    traces = [_make_trace()]
    trace_file = tmp_path / "traces.jsonl"
    _write_traces(trace_file, traces)

    evaluator = TraceEvaluator(trace_file)
    results = evaluator.evaluate()

    out = tmp_path / "results" / "eval.json"
    evaluator.export_results(results, out)

    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert "latency" in data
    assert "n_traces" in data


# ---------------------------------------------------------------------------
# 9. SystemComparison — compare returns correct structure
# ---------------------------------------------------------------------------

def test_system_comparison_compare():
    sys_results = {
        "n_traces": 10,
        "latency": {"mean": 120, "p95": 200},
        "hallucination": {"hallucination_rate": 0.1},
        "bert_score": {"f1": 0.85},
    }
    base_results = {
        "n_traces": 10,
        "latency": {"mean": 150, "p95": 250},
        "hallucination": {"hallucination_rate": 0.2},
        "bert_score": {"f1": 0.75},
    }
    comp = SystemComparison(sys_results, base_results)
    result = comp.compare()

    assert result["n_system_traces"] == 10
    assert result["latency"]["system_mean_ms"] == 120
    assert result["latency"]["baseline_mean_ms"] == 150
    assert result["hallucination"]["system_rate"] == 0.1
    assert result["bert_score"]["delta_f1"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 10. SystemComparison — export writes JSON
# ---------------------------------------------------------------------------

def test_system_comparison_export(tmp_path):
    sys_results = {
        "n_traces": 5,
        "latency": {"mean": 100, "p95": 180},
        "hallucination": {"hallucination_rate": 0.05},
    }
    base_results = {
        "n_traces": 5,
        "latency": {"mean": 130, "p95": 210},
        "hallucination": {"hallucination_rate": 0.15},
    }
    comp = SystemComparison(sys_results, base_results)
    out = tmp_path / "comparison.json"
    comp.export(out)

    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert "latency" in data
    assert "hallucination" in data


# ---------------------------------------------------------------------------
# 11. BERTScore fallback — returns zeros when bert_score not importable
# ---------------------------------------------------------------------------

def test_bert_score_fallback():
    """When bert_score library is not importable, compute_bert_score returns zeros."""
    with patch.dict("sys.modules", {"bert_score": None}):
        result = compute_bert_score(
            ["Hello world"], ["Hello world"], model_name="dummy"
        )
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
