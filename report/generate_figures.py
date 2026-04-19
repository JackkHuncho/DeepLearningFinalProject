"""Generate all poster figures for the F1 Commentary project."""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = Path(__file__).parent.parent

# Consistent color palette
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
ORANGE = "#EA580C"
ORANGE_LIGHT = "#FDBA74"
TEAL = "#0D9488"
TEAL_LIGHT = "#5EEAD4"
GRAY = "#6B7280"
RED = "#DC2626"
GREEN = "#16A34A"

COLORS = ["#2563EB", "#0D9488", "#7C3AED", "#EA580C", "#DC2626",
          "#16A34A", "#D97706", "#4F46E5", "#0891B2", "#BE185D",
          "#65A30D", "#9333EA"]


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {name}.png and {name}.pdf")


# ─────────────────────────────────────────────────────────────────────
# 1. Training Loss Curve
# ─────────────────────────────────────────────────────────────────────
def plot_training_loss():
    print("Generating training loss curve...")
    steps = [5, 10, 15, 20, 25, 30]
    losses = [2.807358, 2.185159, 1.350019, 1.149365, 0.923337, 0.764455]
    epochs = [0.48, 0.95, 1.38, 1.86, 2.29, 2.76]
    eval_step = 33
    eval_loss = 1.181118

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, losses, "o-", color=BLUE, linewidth=2.5, markersize=8,
            label="Training Loss", zorder=5)
    ax.plot(eval_step, eval_loss, "D", color=RED, markersize=12,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"Validation Loss ({eval_loss:.3f})", zorder=6)
    ax.axhline(y=eval_loss, color=RED, linestyle="--", alpha=0.4, linewidth=1)

    # Epoch annotations
    for s, e in zip(steps, epochs):
        if e in (0.95, 1.86, 2.76):
            ax.annotate(f"Epoch {int(round(e))}", (s, losses[steps.index(s)]),
                        textcoords="offset points", xytext=(0, 15),
                        fontsize=8, color=GRAY, ha="center")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SFT Fine-Tuning Loss (Qwen2.5-7B + LoRA)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 36)
    ax.set_ylim(0.5, 3.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Annotation: loss reduction
    ax.annotate("72.8% reduction", xy=(30, 0.764), xytext=(22, 1.6),
                fontsize=10, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    save(fig, "training_loss_curve")


# ─────────────────────────────────────────────────────────────────────
# 2. System Architecture Diagram
# ─────────────────────────────────────────────────────────────────────
def plot_architecture():
    print("Generating system architecture diagram...")
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.5, 4)
    ax.axis("off")

    stages = [
        ("Replay\nEngine", "FastF1\ntelemetry", "#DBEAFE", BLUE),
        ("Recognition\nEngine", "Pattern\ndetection", "#CCFBF1", TEAL),
        ("Beat\nScheduler", "Priority\nselection", "#EDE9FE", "#7C3AED"),
        ("Commentary\nGenerator", "Qwen2.5-7B\n+ LoRA", "#FEF3C7", "#D97706"),
        ("Grounding\nGuard", "Factual\nvalidation", "#FEE2E2", RED),
        ("Output", "Final\ncommentary", "#DCFCE7", GREEN),
    ]

    box_w, box_h = 2.0, 1.6
    gap = 0.55
    y_center = 1.8

    for i, (title, subtitle, bg_color, border_color) in enumerate(stages):
        x = i * (box_w + gap)
        bbox = FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.15",
                              facecolor=bg_color, edgecolor=border_color,
                              linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(x + box_w / 2, y_center + 0.15, title,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=border_color)
        ax.text(x + box_w / 2, y_center - 0.45, subtitle,
                ha="center", va="center", fontsize=8.5, color=GRAY)

        # Arrow to next
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap, y_center),
                        xytext=(x + box_w + 0.05, y_center),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                        lw=2, mutation_scale=18))

    # Retrieval Memory branch (below generator, stage index 3)
    mem_x = 3 * (box_w + gap) + box_w / 2 - box_w / 2
    mem_y = -0.8
    mem_bbox = FancyBboxPatch((mem_x, mem_y - box_h * 0.4), box_w, box_h * 0.7,
                              boxstyle="round,pad=0.12",
                              facecolor="#F0F9FF", edgecolor="#0891B2",
                              linewidth=2, linestyle="--")
    ax.add_patch(mem_bbox)
    ax.text(mem_x + box_w / 2, mem_y + 0.0, "Retrieval Memory",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#0891B2")
    ax.text(mem_x + box_w / 2, mem_y - 0.35, "FAISS + Sentence\nTransformers",
            ha="center", va="center", fontsize=8, color=GRAY)

    # Arrow from memory to generator
    ax.annotate("", xy=(mem_x + box_w / 2, y_center - box_h / 2),
                xytext=(mem_x + box_w / 2, mem_y + box_h * 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#0891B2",
                                lw=1.8, linestyle="--", mutation_scale=15))

    ax.set_title("System Architecture: F1 Commentary Generation Pipeline",
                 fontsize=15, fontweight="bold", pad=15)

    save(fig, "system_architecture")


# ─────────────────────────────────────────────────────────────────────
# 3. Event Type Distribution (Training Data)
# ─────────────────────────────────────────────────────────────────────
def plot_event_distribution():
    print("Generating event type distribution...")
    sft_path = PROJECT / "data" / "datasets" / "sft_train.jsonl"
    categories = {}
    keywords = {
        "DRS Battle": ["DRS"],
        "Pit Strategy": ["pitting", "PIT", "BOX BOX", "pit stop"],
        "Safety Car": ["SAFETY CAR", "RED FLAG", "VSC"],
        "Tire Degradation": ["degradation", "blistering", "tire age", "graining"],
        "Lead Battle": ["P1", "P2", "leads"],
        "Race Start": ["LIGHTS OUT", "Lap 1"],
        "Final Lap": ["FINAL LAP", "last lap"],
        "Weather": ["rain", "wet", "Rain"],
        "Team Radio": ["Team radio", "radio"],
        "Championship": ["champion", "championship"],
        "Recovery Drive": ["recovery", "started P1"],
        "Midfield Battle": ["P6", "P7", "P8", "P9", "P10", "midfield"],
    }

    with open(sft_path) as f:
        for line in f:
            ex = json.loads(line)
            inp = ex.get("input", "") + " " + ex.get("output", "")
            matched = False
            for cat, kws in keywords.items():
                if any(kw in inp for kw in kws):
                    categories[cat] = categories.get(cat, 0) + 1
                    matched = True
                    break
            if not matched:
                categories["Other"] = categories.get("Other", 0) + 1

    # Sort by count
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    names = [c[0] for c in sorted_cats]
    counts = [c[1] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(names[::-1], counts[::-1], color=COLORS[:len(names)],
                   edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold", color=GRAY)

    ax.set_xlabel("Number of Examples", fontsize=12)
    ax.set_title("Training Dataset: Event Type Distribution (105 Examples)",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) + 3)

    save(fig, "event_type_distribution")


# ─────────────────────────────────────────────────────────────────────
# 4. Pipeline Funnel
# ─────────────────────────────────────────────────────────────────────
def plot_pipeline_funnel():
    print("Generating pipeline funnel...")
    summary_path = PROJECT / "code" / "data" / "artifacts" / "sample_run_local" / "run_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    counters = summary.get("counters", {})
    stages = [
        ("Replay Frames", counters.get("frames_processed", 20)),
        ("Candidate Events", counters.get("candidate_events", 103)),
        ("Beats Scheduled", counters.get("beats_selected", 7)),
        ("Commentaries\nGenerated", counters.get("commentaries_generated", 6)),
        ("Finals Emitted", counters.get("finals_emitted", 6)),
    ]

    names = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors_funnel = [BLUE, TEAL, "#7C3AED", "#D97706", GREEN]
    bars = ax.bar(names, values, color=colors_funnel, edgecolor="white",
                  linewidth=1.5, width=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(val), ha="center", fontsize=13, fontweight="bold", color=GRAY)

    # Add filtering annotations
    for i in range(len(values) - 1):
        if values[i] > 0 and values[i + 1] > 0:
            rate = values[i + 1] / values[i] * 100
            mid_x = (bars[i].get_x() + bars[i].get_width() / 2 +
                     bars[i + 1].get_x() + bars[i + 1].get_width() / 2) / 2
            mid_y = max(values[i], values[i + 1]) / 2
            ax.annotate(f"{rate:.0f}%", (mid_x, mid_y),
                        fontsize=9, color=GRAY, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=GRAY, alpha=0.8))

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Pipeline Filtering: From Telemetry to Commentary",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.15)

    save(fig, "pipeline_funnel")


# ─────────────────────────────────────────────────────────────────────
# 5. System vs Baseline Comparison
# ─────────────────────────────────────────────────────────────────────
def plot_system_vs_baseline():
    print("Generating system vs baseline comparison...")
    scenarios = {
        "Lead\nBattle": ("demo_lead_battle", 1, 0),
        "Pit\nStrategy": ("demo_pit_strategy", 1, 0),
        "Race\nControl": ("demo_race_control", 1, 0),
        "Full Sample\nRun": ("sample_run", 6, 0),
    }

    # Try to read actual data
    art_dir = PROJECT / "code" / "data" / "artifacts"
    names = []
    sys_vals = []
    base_vals = []

    for label, (prefix, default_sys, default_base) in scenarios.items():
        names.append(label)
        local_path = art_dir / f"{prefix}_local" / "run_summary.json"
        base_path = art_dir / f"{prefix}_baseline" / "run_summary.json"

        if local_path.exists():
            with open(local_path) as f:
                d = json.load(f)
            sys_vals.append(d.get("counters", {}).get("finals_emitted", default_sys))
        else:
            sys_vals.append(default_sys)

        if base_path.exists():
            with open(base_path) as f:
                d = json.load(f)
            base_vals.append(d.get("counters", {}).get("finals_emitted", default_base))
        else:
            base_vals.append(default_base)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, sys_vals, width, label="System (Fine-tuned)",
                   color=BLUE, edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width / 2, base_vals, width, label="Baseline",
                   color=ORANGE_LIGHT, edgecolor="white", linewidth=1)

    for bar, val in zip(bars1, sys_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", fontsize=12, fontweight="bold", color=BLUE)
    for bar, val in zip(bars2, base_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", fontsize=12, fontweight="bold", color=ORANGE)

    ax.set_ylabel("Final Commentaries Emitted", fontsize=12)
    ax.set_title("System vs Baseline: Commentary Output by Scenario",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(sys_vals), max(base_vals)) * 1.3)

    save(fig, "system_vs_baseline")


# ─────────────────────────────────────────────────────────────────────
# 6. LoRA Config Summary
# ─────────────────────────────────────────────────────────────────────
def plot_lora_config():
    print("Generating LoRA config summary...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    rows = [
        ["Base Model", "Qwen2.5-7B (4-bit quantized)"],
        ["LoRA Rank (r)", "16"],
        ["LoRA Alpha (α)", "32"],
        ["Target Modules", "q, k, v, o, gate, up, down"],
        ["Trainable Params", "40.4M / 7.66B (0.53%)"],
        ["Epochs", "3"],
        ["Batch Size", "1 (× 8 grad accum)"],
        ["Learning Rate", "2 × 10⁻⁴ (cosine decay)"],
        ["Dataset", "84 train / 21 validation"],
        ["Final Train Loss", "0.764"],
        ["Final Eval Loss", "1.181"],
        ["Training Time", "~2 minutes (RTX 4060)"],
    ]

    col_labels = ["Parameter", "Value"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc="left", loc="center",
                     colWidths=[0.4, 0.55])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        cell.set_edgecolor("white")

    # Style rows
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_facecolor("#F8FAFC" if i % 2 == 0 else "white")
            cell.set_edgecolor("#E2E8F0")
            if j == 0:
                cell.set_text_props(fontweight="bold", color="#374151")
            else:
                cell.set_text_props(color="#1F2937")

    ax.set_title("LoRA Fine-Tuning Configuration",
                 fontsize=15, fontweight="bold", pad=20)

    save(fig, "lora_config_summary")


# ─────────────────────────────────────────────────────────────────────
# 7. Qualitative Comparison
# ─────────────────────────────────────────────────────────────────────
def plot_qualitative_comparison():
    print("Generating qualitative comparison...")
    examples = [
        {
            "scenario": "Lead Battle — VER vs HAM, Lap 52/56, Monza",
            "system": (
                "The gap is 1.2 seconds between VER and HAM and both drivers on "
                "different strategies — Verstappen on HARD while Hamilton sticks with "
                "MEDIUM. The DRS flap opens next lap, which means Hamilton can close "
                "the gap to within a second in the next sector. If Verstappen makes a "
                "mistake or Hamilton finds a run through the medium-speed corners, "
                "this could be over in three laps."
            ),
            "baseline": (
                "At the front of the field, VER continues to pressure PER with a "
                "gap that suggests a potential challenge for position in the coming "
                "laps."
            ),
        },
        {
            "scenario": "Pit Strategy — LEC pits, Lap 45/70, Hungary",
            "system": (
                "Leclerc dives into the pits and Ferrari are going aggressive — soft "
                "tires for the final twenty-five laps! That is a bold call at Budapest "
                "in these temperatures. The soft compound typically gives you eighteen "
                "to twenty laps here, so Charles is going to need to build a big enough "
                "advantage before the degradation cliff hits."
            ),
            "baseline": (
                "LEC has committed to a pit stop, rolling the dice on strategy with "
                "a switch to fresh rubber."
            ),
        },
        {
            "scenario": "Safety Car — Multi-car incident, Lap 22/58, Melbourne",
            "system": (
                "Safety car! Stroll has stopped at turn nine, the Aston Martin off the "
                "track and that brings out the safety car. This completely reshapes the "
                "race — Verstappen's five-point-eight-second lead is gone in an instant. "
                "Now the critical question: does Max pit?"
            ),
            "baseline": (
                "Race control intervenes — the safety car is deployed."
            ),
        },
    ]

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, len(examples) * 4.2))
    if len(examples) == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples):
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Scenario header
        ax.add_patch(FancyBboxPatch((0.1, 8.2), 9.8, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5))
        ax.text(5, 8.95, ex["scenario"], ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1E293B")

        # System output (left)
        ax.add_patch(FancyBboxPatch((0.1, 0.3), 4.75, 7.5, boxstyle="round,pad=0.15",
                                     facecolor="#EFF6FF", edgecolor=BLUE, linewidth=2))
        ax.text(2.475, 7.3, "System (Fine-tuned)", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLUE)
        wrapped = textwrap.fill(ex["system"], width=42)
        ax.text(0.4, 6.7, wrapped, ha="left", va="top",
                fontsize=9, color="#1E293B", linespacing=1.4)

        # Baseline output (right)
        ax.add_patch(FancyBboxPatch((5.15, 0.3), 4.75, 7.5, boxstyle="round,pad=0.15",
                                     facecolor="#FFF7ED", edgecolor=ORANGE, linewidth=2))
        ax.text(7.525, 7.3, "Baseline (Template)", ha="center", va="center",
                fontsize=11, fontweight="bold", color=ORANGE)
        wrapped = textwrap.fill(ex["baseline"], width=42)
        ax.text(5.45, 6.7, wrapped, ha="left", va="top",
                fontsize=9, color="#1E293B", linespacing=1.4)

    fig.suptitle("Qualitative Comparison: System vs Baseline Commentary",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.15)

    save(fig, "qualitative_comparison")


# ─────────────────────────────────────────────────────────────────────
# 8. Model Parameter Efficiency
# ─────────────────────────────────────────────────────────────────────
def plot_param_efficiency():
    print("Generating parameter efficiency chart...")
    fig, ax = plt.subplots(figsize=(7, 5))

    total = 7655.986688  # millions
    trainable = 40.370176
    frozen = total - trainable

    wedges, texts, autotexts = ax.pie(
        [trainable, frozen],
        labels=["LoRA Trainable\n(40.4M)", "Frozen\n(7,616M)"],
        colors=[BLUE, "#E2E8F0"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        explode=(0.08, 0),
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    autotexts[0].set_fontweight("bold")
    autotexts[0].set_color("white")
    autotexts[1].set_color(GRAY)

    ax.set_title("Parameter Efficiency: LoRA Fine-Tuning\nQwen2.5-7B (4-bit)",
                 fontsize=14, fontweight="bold")

    save(fig, "param_efficiency")


# ─────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Generating poster figures...")
    print("=" * 60)

    plot_training_loss()
    plot_architecture()
    plot_event_distribution()
    plot_pipeline_funnel()
    plot_system_vs_baseline()
    plot_lora_config()
    plot_qualitative_comparison()
    plot_param_efficiency()

    print("=" * 60)
    print(f"All figures saved to {FIGURES_DIR}")
    print("=" * 60)
