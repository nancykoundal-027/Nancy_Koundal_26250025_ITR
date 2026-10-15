from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

RESULTS = Path("results")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

DATASETS = [
    ("Baseline (±2 cm)", RESULTS / "heal_batch_episodes_25.csv"),
    ("Wide stress (±12 cm)", RESULTS / "heal_wide_stress_episodes_15.csv"),
]
METHODS = ["Mink", "DLS-IK", "QP-IK"]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required results file missing: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


datasets = []
for label, path in DATASETS:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No episode rows found in {path}")
    datasets.append((label, rows))
    print(f"Loaded {len(rows)} episodes from {path}")

# 1. Success rate comparison
conditions = [label for label, _ in datasets]
x = list(range(len(METHODS)))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5.5))

for i, (label, rows) in enumerate(datasets):
    rates = []
    for method in METHODS:
        group = [r for r in rows if r["Method"] == method]
        rate = (
            100 * sum(as_bool(r["Success"]) for r in group) / len(group)
            if group else 0
        )
        rates.append(rate)

    offsets = [v + (i - 0.5) * width for v in x]
    bars = ax.bar(offsets, rates, width, label=label)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)
ax.set_ylim(0, 115)
ax.set_ylabel("Successful episodes (%)")
ax.set_title("HEAL Pick-and-Place: Success Rate")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(rect=[0, 0.13, 1, 1])
fig.savefig(PLOTS / "heal_success_rate_comparison.png", dpi=200)
plt.close(fig)

# 2. Wall runtime distributions
fig, ax = plt.subplots(figsize=(10, 5.5))
box_data = []
box_labels = []

for label, rows in datasets:
    for method in METHODS:
        vals = [
            as_float(r.get("Wall_runtime_s", ""))
            for r in rows if r["Method"] == method
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            box_data.append(vals)
            box_labels.append(f"{method}\n{label}")

if box_data:
    ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)
    ax.set_ylabel("Wall runtime (seconds)")
    ax.set_title("Episode Runtime Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "heal_runtime_comparison.png", dpi=200)
else:
    print("WARNING: No valid runtime values; runtime plot skipped.")
plt.close(fig)

# 3. Final XY distance from tray centre for verified placements
fig, ax = plt.subplots(figsize=(10, 5.5))
placement_data = []
placement_labels = []

for label, rows in datasets:
    for method in METHODS:
        vals = [
            as_float(r.get("Final_cube_xy_error_from_tray_centre_m", ""))
            for r in rows
            if r["Method"] == method and as_bool(r.get("Success", False))
        ]
        vals = [v * 100 for v in vals if v is not None]
        if vals:
            placement_data.append(vals)
            placement_labels.append(f"{method}\n{label}")

if placement_data:
    ax.boxplot(placement_data, tick_labels=placement_labels, showmeans=True)
    ax.set_ylabel("Final XY distance from tray centre (cm)")
    ax.set_title("Placement Accuracy in Successful Episodes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "heal_placement_accuracy.png", dpi=200)
else:
    print("WARNING: No valid placement-error values; accuracy plot skipped.")
plt.close(fig)

# 4. Save a concise summary for the report
summary = [
    "# HEAL Evaluation Summary",
    "",
    "Success rates from the recorded baseline and wide stress tests.",
    "The stress test varied initial cube x/y position by ±12 cm; grasp "
    "targets were shifted with the cube, so it is not a fixed-target test.",
    "",
    "| Test | Method | Successes | Episodes | Success rate |",
    "|---|---|---:|---:|---:|",
]

for label, rows in datasets:
    for method in METHODS:
        group = [r for r in rows if r["Method"] == method]
        successes = sum(as_bool(r["Success"]) for r in group)
        n = len(group)
        rate = 100 * successes / n if n else 0
        summary.append(
            f"| {label} | {method} | {successes} | {n} | {rate:.1f}% |"
        )

summary += [
    "",
    "## Interpretation notes",
    "",
    "- The baseline contains 25 episodes per method.",
    "- The wide stress test contains 15 episodes per method.",
    "- These tests vary starting positions but shift the grasp targets "
    "with the cube; results therefore do not measure fixed-target "
    "robustness.",
    "- Wall runtime includes script setup and computation, and excludes "
    "viewer rendering and real-time sleeps.",
    "- Mink uses joint-space interpolation between IK configurations, "
    "while DLS and QP use Cartesian velocity tracking. Runtime comparisons "
    "therefore reflect the complete implementations, not only solver speed.",
    "- Successful-placement accuracy excludes failed episodes.",
]
(RESULTS / "heal_evaluation_summary.md").write_text(
    "\n".join(summary) + "\n"
)

print("\nCreated plot files:")
for name in [
    "heal_success_rate_comparison.png",
    "heal_runtime_comparison.png",
    "heal_placement_accuracy.png",
]:
    p = PLOTS / name
    if p.exists():
        print(" ", p)

print("\nCreated:", RESULTS / "heal_evaluation_summary.md")
