"""Evaluation metrics: BERTScore, human eval, latency, hallucination rate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BERTScoreResult:
    precision: float
    recall: float
    f1: float


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    model_name: str = "microsoft/deberta-xlarge-mnli",
) -> BERTScoreResult:
    """Compute BERTScore between predictions and references.

    Uses bert-score library. Falls back to dummy scores if library unavailable.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type=model_name,
            verbose=False,
        )
        return BERTScoreResult(
            precision=P.mean().item(),
            recall=R.mean().item(),
            f1=F1.mean().item(),
        )
    except ImportError:
        logger.warning("bert-score not installed, returning dummy scores")
        return BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)


def compute_latency_metrics(latencies_ms: list[float]) -> dict:
    """Compute latency statistics."""
    import statistics
    if not latencies_ms:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    sorted_l = sorted(latencies_ms)
    p95_idx = int(len(sorted_l) * 0.95)
    p99_idx = int(len(sorted_l) * 0.99)
    return {
        "mean": statistics.mean(sorted_l),
        "median": statistics.median(sorted_l),
        "p95": sorted_l[min(p95_idx, len(sorted_l) - 1)],
        "p99": sorted_l[min(p99_idx, len(sorted_l) - 1)],
        "max": max(sorted_l),
    }


def compute_hallucination_rate(traces: list[dict]) -> dict:
    """Compute hallucination rate from guard decisions in traces.

    Uses guard_decision.claims to count unsupported claims.
    """
    total_claims = 0
    unsupported = 0
    for trace in traces:
        guard = trace.get("guard_decision", {})
        claims = guard.get("claims", [])
        for claim in claims:
            total_claims += 1
            if claim.get("support_type") == "unsupported":
                unsupported += 1
    rate = unsupported / total_claims if total_claims > 0 else 0.0
    return {
        "total_claims": total_claims,
        "unsupported_claims": unsupported,
        "hallucination_rate": rate,
    }


@dataclass
class HumanEvalForm:
    """Human evaluation form for a single commentary example."""
    example_id: str
    system_output: str
    baseline_output: str
    race_context: str
    # Likert scales 1-5
    fluency_system: Optional[int] = None
    fluency_baseline: Optional[int] = None
    accuracy_system: Optional[int] = None
    accuracy_baseline: Optional[int] = None
    insight_system: Optional[int] = None
    insight_baseline: Optional[int] = None
    preference: Optional[str] = None  # "system", "baseline", "tie"
    notes: Optional[str] = None


def generate_human_eval_forms(
    system_outputs: list[dict],
    baseline_outputs: list[dict],
    output_path: Path,
) -> list[HumanEvalForm]:
    """Generate JSON evaluation forms for human reviewers."""
    forms = []
    for i, (sys_out, base_out) in enumerate(zip(system_outputs, baseline_outputs)):
        form = HumanEvalForm(
            example_id=f"eval_{i:03d}",
            system_output=sys_out.get("text", ""),
            baseline_output=base_out.get("text", ""),
            race_context=sys_out.get("context", ""),
        )
        forms.append(form)

    # Write forms to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([vars(form) for form in forms], f, indent=2)

    return forms


def aggregate_human_eval(forms_path: Path) -> dict:
    """Aggregate completed human eval forms into summary metrics."""
    with open(forms_path) as f:
        forms = json.load(f)

    metrics = {
        "n_evaluated": 0,
        "avg_fluency_system": 0.0,
        "avg_fluency_baseline": 0.0,
        "avg_accuracy_system": 0.0,
        "avg_accuracy_baseline": 0.0,
        "avg_insight_system": 0.0,
        "avg_insight_baseline": 0.0,
        "preference_system": 0,
        "preference_baseline": 0,
        "preference_tie": 0,
    }

    evaluated = [f for f in forms if f.get("fluency_system") is not None]
    if not evaluated:
        return metrics

    n = len(evaluated)
    metrics["n_evaluated"] = n
    for key in ["fluency", "accuracy", "insight"]:
        for variant in ["system", "baseline"]:
            field = f"{key}_{variant}"
            avg = sum(f.get(field, 0) or 0 for f in evaluated) / n
            metrics[f"avg_{field}"] = round(avg, 2)

    for f in evaluated:
        pref = f.get("preference", "tie")
        metrics[f"preference_{pref}"] = metrics.get(f"preference_{pref}", 0) + 1

    return metrics
