"""Trace-based evaluation runner."""

import json
from pathlib import Path
from typing import Optional

from f1_commentary.evaluation.metrics import (
    compute_bert_score,
    compute_latency_metrics,
    compute_hallucination_rate,
    BERTScoreResult,
)


class TraceEvaluator:
    """Evaluate system performance from logged traces."""

    def __init__(self, trace_path: Path):
        self.traces = self._load_traces(trace_path)

    @staticmethod
    def _load_traces(path: Path) -> list[dict]:
        traces = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces

    def evaluate(self, references: Optional[list[str]] = None) -> dict:
        """Run all metrics on loaded traces."""
        results = {}

        # Latency
        latencies = [t.get("total_latency_ms", 0) for t in self.traces]
        results["latency"] = compute_latency_metrics(latencies)

        # Hallucination
        results["hallucination"] = compute_hallucination_rate(self.traces)

        # BERTScore (if references provided)
        if references:
            predictions = []
            for t in self.traces:
                fc = t.get("final_commentary", {})
                text = fc.get("final_text", "") if isinstance(fc, dict) else ""
                predictions.append(text)

            # Align lengths
            min_len = min(len(predictions), len(references))
            bert = compute_bert_score(predictions[:min_len], references[:min_len])
            results["bert_score"] = {
                "precision": bert.precision,
                "recall": bert.recall,
                "f1": bert.f1,
            }

        results["n_traces"] = len(self.traces)
        return results

    def export_results(self, results: dict, output_path: Path) -> None:
        """Save evaluation results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
