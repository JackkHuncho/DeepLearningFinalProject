"""Side-by-side system vs baseline comparison."""

import json
from pathlib import Path


class SystemComparison:
    """Compare full system against baseline."""

    def __init__(self, system_results: dict, baseline_results: dict):
        self.system = system_results
        self.baseline = baseline_results

    def compare(self) -> dict:
        """Generate comparison summary."""
        comparison = {
            "n_system_traces": self.system.get("n_traces", 0),
            "n_baseline_traces": self.baseline.get("n_traces", 0),
        }

        # Latency comparison
        sys_lat = self.system.get("latency", {})
        base_lat = self.baseline.get("latency", {})
        comparison["latency"] = {
            "system_mean_ms": sys_lat.get("mean", 0),
            "baseline_mean_ms": base_lat.get("mean", 0),
            "system_p95_ms": sys_lat.get("p95", 0),
            "baseline_p95_ms": base_lat.get("p95", 0),
        }

        # Hallucination comparison
        sys_hall = self.system.get("hallucination", {})
        base_hall = self.baseline.get("hallucination", {})
        comparison["hallucination"] = {
            "system_rate": sys_hall.get("hallucination_rate", 0),
            "baseline_rate": base_hall.get("hallucination_rate", 0),
        }

        # BERTScore comparison
        sys_bert = self.system.get("bert_score", {})
        base_bert = self.baseline.get("bert_score", {})
        if sys_bert and base_bert:
            comparison["bert_score"] = {
                "system_f1": sys_bert.get("f1", 0),
                "baseline_f1": base_bert.get("f1", 0),
                "delta_f1": sys_bert.get("f1", 0) - base_bert.get("f1", 0),
            }

        return comparison

    def export(self, output_path: Path) -> None:
        """Save comparison to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.compare(), f, indent=2)
