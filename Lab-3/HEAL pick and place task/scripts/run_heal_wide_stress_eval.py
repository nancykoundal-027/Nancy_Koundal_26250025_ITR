from pathlib import Path
import argparse
import builtins
import contextlib
import csv
import io
import math
import re
import time
import traceback

import numpy as np
import mujoco
import mujoco.viewer


METHODS = [
    ("Mink", Path("scripts/heal_mink_pick_place_standardized.py")),
    ("DLS-IK", Path("scripts/heal_dls_pick_place_smooth.py")),
    ("QP-IK", Path("scripts/heal_qp_pick_place.py")),
]

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class HeadlessViewer:
    """Small viewer substitute so the task physics can run without rendering."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return False

    def is_running(self):
        return (
            not self.state["finished"]
            and self.state["syncs"] < self.state["max_syncs"]
        )

    def sync(self):
        self.state["syncs"] += 1


def patch_episode(source_path, dx, dy):
    """Make an in-memory episode copy; do not edit the source script."""
    src = source_path.read_text()

    data_line = "data = mujoco.MjData(model)"
    if data_line not in src:
        raise RuntimeError(
            f"{source_path}: could not find the initial MjData creation."
        )

    # Randomize the cube's free-joint translation for this episode.
    # The same dx, dy will be added to the grasp-side targets below.
    injection = f"""
# Reproducible benchmark episode initialization
_eval_cube_body = int(model.body("cube").id)
_eval_cube_joint = int(model.body_jntadr[_eval_cube_body])
if _eval_cube_joint < 0:
    raise RuntimeError("Cube does not have a joint.")
_eval_cube_qadr = int(model.jnt_qposadr[_eval_cube_joint])
data.qpos[_eval_cube_qadr:_eval_cube_qadr + 3] = np.array(
    [{0.45 + dx:.8f}, {0.10 + dy:.8f}, 0.225]
)
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
"""
    src = src.replace(data_line, data_line + injection, 1)

    # Apply precisely the same XY shift to the cube approach, grasp and lift.
    target_pattern = re.compile(
        r"^TRAY_PLACE\s*=\s*np\.array\(\[[^\]]+\]\)\s*$",
        re.MULTILINE,
    )
    match = target_pattern.search(src)
    if not match:
        raise RuntimeError(
            f"{source_path}: could not find TRAY_PLACE; source unchanged."
        )

    target_injection = f"""

# Matched cube-location offset for the benchmark
_EVAL_SHIFT = np.array([{dx:.8f}, {dy:.8f}])
PREGRASP[:2] += _EVAL_SHIFT
GRASP[:2] += _EVAL_SHIFT
LIFT[:2] += _EVAL_SHIFT
"""
    src = src[:match.end()] + target_injection + src[match.end():]
    return src


def last_float(pattern, output):
    matches = re.findall(pattern, output, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def extract_metrics(output):
    success = "RESULT: PICK-AND-PLACE SUCCESS" in output
    lift_verified = bool(re.search(
        r"Lift verified:\s*True", output, re.IGNORECASE
    ))
    place_verified = bool(re.search(
        r"Place verified:\s*True", output, re.IGNORECASE
    ))

    rise_pattern = (
        r"(?:Vertical rise(?:\s*\(m\))?|"
        r"Cube vertical displacement\s*\(m\)):\s*("
        + NUMBER + ")"
    )
    lateral_pattern = (
        r"Lateral shift(?:\s*\(m\))?:\s*(" + NUMBER + ")"
    )

    rise = last_float(rise_pattern, output)
    lateral = last_float(lateral_pattern, output)

    final_positions = re.findall(
        r"Final cube position:\s*\[([^\]]+)\]", output
    )
    xyz = [None, None, None]
    placement_error = None

    if final_positions:
        values = [float(x) for x in re.findall(NUMBER, final_positions[-1])]
        if len(values) >= 3:
            xyz = values[:3]
            # XY distance from the tray centre: x=0.35 m, y=-0.15 m.
            placement_error = math.hypot(xyz[0] - 0.35, xyz[1] + 0.15)

    if success:
        status = "success"
    elif "RESULT:" in output:
        status = "task_failed_or_not_verified"
    else:
        status = "error_or_missing_summary"

    return {
        "Success": success,
        "Lift_verified": lift_verified,
        "Place_verified": place_verified,
        "Vertical_rise_m": rise,
        "Lateral_shift_during_lift_m": lateral,
        "Final_cube_x_m": xyz[0],
        "Final_cube_y_m": xyz[1],
        "Final_cube_z_m": xyz[2],
        "Final_cube_xy_error_from_tray_centre_m": placement_error,
        "Status": status,
    }


def run_one(method, source_path, episode, seed, dx, dy, log_dir):
    state = {"finished": False, "syncs": 0, "max_syncs": 250000}
    output_buffer = io.StringIO()

    source_text = patch_episode(source_path, dx, dy)
    source_label = str(source_path.resolve())

    original_print = builtins.print
    original_sleep = time.sleep
    original_launch = mujoco.viewer.launch_passive

    def tracked_print(*args, **kwargs):
        message = kwargs.get("sep", " ").join(str(x) for x in args)
        if "RESULT:" in message:
            state["finished"] = True
        original_print(*args, **kwargs)

    def fake_launch(model, data):
        return HeadlessViewer(state)

    start = time.perf_counter()

    try:
        with contextlib.redirect_stdout(output_buffer), \
             contextlib.redirect_stderr(output_buffer):
            builtins.print = tracked_print
            time.sleep = lambda *_args, **_kwargs: None
            mujoco.viewer.launch_passive = fake_launch

            try:
                namespace = {
                    "__name__": "__main__",
                    "__file__": source_label,
                }
                exec(
                    compile(source_text, source_label, "exec"),
                    namespace,
                )
            except Exception:
                traceback.print_exc()
    finally:
        builtins.print = original_print
        time.sleep = original_sleep
        mujoco.viewer.launch_passive = original_launch

    elapsed = time.perf_counter() - start
    output = output_buffer.getvalue()

    log_path = log_dir / (
        f"{method.lower().replace('-', '')}_ep{episode:02d}_seed{seed}.log"
    )
    log_path.write_text(output)

    metrics = extract_metrics(output)
    row = {
        "Method": method,
        "Episode": episode,
        "Seed": seed,
        "Cube_start_x_m": 0.45 + dx,
        "Cube_start_y_m": 0.10 + dy,
        "Wall_runtime_s": round(elapsed, 4),
        **metrics,
        "Log_file": str(log_path),
    }

    original_print(
        f"{method:6s} episode {episode:02d}: "
        f"success={row['Success']}, "
        f"lift={row['Lift_verified']}, "
        f"place={row['Place_verified']}, "
        f"status={row['Status']}"
    )

    if row["Status"] != "success":
        original_print(f"  Inspect log: {log_path}")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1.")

    for _, path in METHODS:
        if not path.exists():
            raise SystemExit(f"Missing script: {path}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    log_dir = results_dir / f"heal_wide_stress_logs_{args.episodes}"
    log_dir.mkdir(exist_ok=True)

    # Precompute offsets so every method sees identical starts per episode.
    episodes = []
    for index in range(args.episodes):
        seed = args.base_seed + index
        rng = np.random.default_rng(seed)
        dx, dy = rng.uniform(-0.12, 0.12, size=2)
        episodes.append((index + 1, seed, float(dx), float(dy)))

    rows = []
    print(f"Running {args.episodes} episode(s) per method.")
    print("Headless physics: rendering and real-time sleeps are disabled.")
    print("Cube offsets are matched across methods for each episode.")
    print("Wall runtime includes script setup and controller computation.")

    for method, source_path in METHODS:
        for episode, seed, dx, dy in episodes:
            row = run_one(
                method, source_path, episode, seed, dx, dy, log_dir
            )
            rows.append(row)

    csv_path = results_dir / f"heal_wide_stress_episodes_{args.episodes}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = results_dir / (
        f"heal_wide_stress_summary_{args.episodes}.md"
    )
    lines = [
        "# HEAL Headless IK Evaluation",
        "",
        f"Episodes per method: {args.episodes}",
        f"Base seed: {args.base_seed}",
        "The same cube offsets are used across methods for each episode.",
        "Wall runtime includes setup and computation, but excludes rendering "
        "and real-time sleeping.",
        "",
        "| Method | Successes | Success rate | Lift verified | Place verified | "
        "Mean wall runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method, _ in METHODS:
        group = [r for r in rows if r["Method"] == method]
        n = len(group)
        successes = sum(bool(r["Success"]) for r in group)
        lifts = sum(bool(r["Lift_verified"]) for r in group)
        places = sum(bool(r["Place_verified"]) for r in group)
        mean_runtime = sum(r["Wall_runtime_s"] for r in group) / n

        lines.append(
            f"| {method} | {successes}/{n} | "
            f"{100.0 * successes / n:.1f}% | "
            f"{lifts}/{n} | {places}/{n} | {mean_runtime:.4f} |"
        )

    lines += [
        "",
        "## Interpretation note",
        "",
        "These are system-level results from the implementations used here. "
        "Mink uses joint-space interpolation between IK configurations, "
        "while DLS and QP use Cartesian velocity tracking, so timing and "
        "trajectory comparisons are affected by controller architecture.",
        "",
        "Do not treat a one-episode pilot as statistical evidence. Run 25 "
        "episodes per method for the main evaluation.",
    ]
    summary_path.write_text("\n".join(lines) + "\n")

    print("\nSaved:", csv_path)
    print("Saved:", summary_path)
    print("Per-episode logs:", log_dir)


if __name__ == "__main__":
    main()
