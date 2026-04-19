"""Paper artifact generation: tables, figures, qualitative examples."""

import json
import csv
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ArtifactExporter:
    """Generate paper-ready artifacts from evaluation results and traces."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.examples_dir = self.output_dir / "qualitative_examples"
        for d in [self.figures_dir, self.tables_dir, self.examples_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def generate_latex_metrics_table(
        self,
        comparison: dict,
        output_name: str = "metrics_comparison.tex",
    ) -> Path:
        r"""Generate a LaTeX table comparing system vs baseline metrics.

        Table includes:
        - Latency (mean, p95)
        - Hallucination rate
        - BERTScore F1 (if available)

        Format as a proper LaTeX tabular that can be \input{} into the report.
        """
        lat = comparison.get("latency", {})
        hall = comparison.get("hallucination", {})
        bert = comparison.get("bert_score", {})

        rows = [
            (
                "Latency — Mean (ms)",
                f"{lat.get('system_mean_ms', 0):.1f}",
                f"{lat.get('baseline_mean_ms', 0):.1f}",
            ),
            (
                "Latency — P95 (ms)",
                f"{lat.get('system_p95_ms', 0):.1f}",
                f"{lat.get('baseline_p95_ms', 0):.1f}",
            ),
            (
                "Hallucination Rate",
                f"{hall.get('system_rate', 0):.1%}",
                f"{hall.get('baseline_rate', 0):.1%}",
            ),
        ]

        if bert:
            rows.append((
                "BERTScore F1",
                f"{bert.get('system_f1', 0):.3f}",
                f"{bert.get('baseline_f1', 0):.3f}",
            ))

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{System vs.\ Baseline Metrics Comparison}",
            r"\label{tab:metrics-comparison}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Metric & Our System & Baseline \\",
            r"\midrule",
        ]

        for metric, sys_val, base_val in rows:
            lines.append(f"{metric} & {sys_val} & {base_val} \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        path = self.tables_dir / output_name
        path.write_text("\n".join(lines) + "\n")
        return path

    def generate_latency_chart(
        self,
        system_latencies: list[float],
        baseline_latencies: list[float],
        output_name: str = "latency_comparison.png",
    ) -> Path:
        """Generate matplotlib latency comparison chart (box plot)."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # Box plot comparing system vs baseline latencies
        ax.boxplot(
            [system_latencies, baseline_latencies],
            tick_labels=["Our System", "Baseline"],
        )
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Commentary Generation Latency")

        path = self.figures_dir / output_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_bert_score_chart(
        self,
        system_scores: list[float],
        baseline_scores: list[float],
        output_name: str = "bertscore_comparison.png",
    ) -> Path:
        """Generate BERTScore F1 distribution comparison chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # Bar chart of mean scores with individual points overlaid
        means = [
            sum(system_scores) / len(system_scores) if system_scores else 0,
            sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        ]
        bars = ax.bar(["Our System", "Baseline"], means, color=["#2196F3", "#FF9800"])
        ax.set_ylabel("BERTScore F1")
        ax.set_title("BERTScore Comparison")
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=12)

        path = self.figures_dir / output_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_hallucination_chart(
        self,
        system_rate: float,
        baseline_rate: float,
        output_name: str = "hallucination_comparison.png",
    ) -> Path:
        """Generate hallucination rate comparison bar chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        bars = ax.bar(
            ["Our System", "Baseline"],
            [system_rate, baseline_rate],
            color=["#4CAF50", "#f44336"],
        )
        ax.set_ylabel("Hallucination Rate")
        ax.set_title("Hallucination Rate Comparison")
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, [system_rate, baseline_rate]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.1%}", ha="center", fontsize=12)

        path = self.figures_dir / output_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_qualitative_examples(
        self,
        examples: list[dict],
        output_name: str = "qualitative_examples.tex",
    ) -> Path:
        r"""Generate LaTeX-formatted qualitative comparison examples.

        Each example dict should have:
        - "context": race situation description
        - "system_output": our system's commentary
        - "baseline_output": baseline's commentary
        - "label": example label/scenario name

        Format as LaTeX figures/tables that can be \input{} into the report.
        """
        lines: list[str] = []

        for i, ex in enumerate(examples, 1):
            label = ex.get("label", f"Example {i}")
            context = _escape_latex(ex.get("context", ""))
            sys_out = _escape_latex(ex.get("system_output", ""))
            base_out = _escape_latex(ex.get("baseline_output", ""))

            lines += [
                r"\begin{table}[ht]",
                r"\centering",
                f"\\caption{{Qualitative Example: {_escape_latex(label)}}}",
                f"\\label{{tab:qual-example-{i}}}",
                r"\begin{tabular}{p{0.15\textwidth} p{0.75\textwidth}}",
                r"\toprule",
                f"\\textbf{{Context}} & {context} \\\\",
                r"\midrule",
                f"\\textbf{{Our System}} & {sys_out} \\\\",
                r"\midrule",
                f"\\textbf{{Baseline}} & {base_out} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]

        path = self.examples_dir / output_name
        path.write_text("\n".join(lines) + "\n")
        return path

    def generate_all(
        self,
        comparison: dict,
        system_latencies: list[float],
        baseline_latencies: list[float],
        system_bert_scores: list[float],
        baseline_bert_scores: list[float],
        examples: list[dict],
    ) -> dict[str, Path]:
        """Generate all artifacts. Returns dict of artifact name -> path."""
        paths = {}
        paths["metrics_table"] = self.generate_latex_metrics_table(comparison)
        paths["latency_chart"] = self.generate_latency_chart(system_latencies, baseline_latencies)
        paths["bertscore_chart"] = self.generate_bert_score_chart(system_bert_scores, baseline_bert_scores)

        sys_hall = comparison.get("hallucination", {}).get("system_rate", 0)
        base_hall = comparison.get("hallucination", {}).get("baseline_rate", 0)
        paths["hallucination_chart"] = self.generate_hallucination_chart(sys_hall, base_hall)

        if examples:
            paths["qualitative_examples"] = self.generate_qualitative_examples(examples)

        return paths


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in user-supplied text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def generate_reproducibility_script(output_path: Path) -> Path:
    """Generate a shell script that reproduces all paper artifacts from saved traces."""
    script = '''#!/usr/bin/env bash
set -euo pipefail

echo "=== Reproducing paper artifacts ==="

# Evaluate system traces
python -m f1_commentary.cli evaluate-run \\
    --traces data/logs/traces/system_traces.jsonl \\
    --output data/artifacts/system_results.json

# Evaluate baseline traces
python -m f1_commentary.cli evaluate-run \\
    --traces data/logs/traces/baseline_traces.jsonl \\
    --output data/artifacts/baseline_results.json

# Compare
python -m f1_commentary.cli compare-system-vs-baseline \\
    --system-traces data/logs/traces/system_traces.jsonl \\
    --baseline-traces data/logs/traces/baseline_traces.jsonl \\
    --output data/artifacts/comparison.json

# Generate artifacts
python -m f1_commentary.cli generate-artifacts \\
    --comparison data/artifacts/comparison.json \\
    --output data/artifacts/paper

echo "=== Done. Artifacts in data/artifacts/paper/ ==="
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)
    output_path.chmod(0o755)
    return output_path
