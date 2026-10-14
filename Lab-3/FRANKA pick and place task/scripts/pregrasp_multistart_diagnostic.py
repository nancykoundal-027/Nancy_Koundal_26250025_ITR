#!/usr/bin/env python3
"""Offline multi-start Mink IK diagnosis for failed pre-grasp poses.

This is a kinematic diagnostic only: it does not step the physical simulation,
check collisions, or claim the robot can execute a collision-free path.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "robot_descriptions" / "franka_emika_panda" / "pick_place_scene.xml"
RESULTS = ROOT / "results"
SOURCE_JSON = RESULTS / "mink_results_collision_free_25_episodes.json"
OUT_CSV = RESULTS / "pregrasp_multistart_diagnostic.csv"
OUT_TXT = RESULTS / "pregrasp_multistart_diagnostic_summary.txt"

DT = 0.002
MAX_ITERS = 1800
POSITION_TOL = 0.010
ORIENTATION_TOL = 0.08
N_RANDOM_STARTS = 3  # plus the model's home configuration
SEED = 14013


def named_id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise RuntimeError(f"Required model object not found: {name}")
    return idx


def main():
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(
            f"Cannot find {SOURCE_JSON}. Check the results filename."
        )

    records = json.loads(SOURCE_JSON.read_text())
    failed = [r for r in records if not r.get("success", False)
              and "Pre-grasp" in str(r.get("reason", ""))]
    if not failed:
        print("No failed Pre-grasp episodes found in the source JSON.")
        return

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = DT
    base_data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, base_data, home_id)
    else:
        mujoco.mj_resetData(model, base_data)
    mujoco.mj_forward(model, base_data)
    base_qpos = base_data.qpos.copy()

    arm_joint_ids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                     for i in range(1, 8)]
    arm_qpos = np.array([model.jnt_qposadr[j] for j in arm_joint_ids], dtype=int)
    arm_ranges = np.array([model.jnt_range[j] for j in arm_joint_ids], dtype=float)
    hand_task = mink.FrameTask(
        frame_name="hand", frame_type="body", position_cost=1.0,
        orientation_cost=1.0, lm_damping=1.0
    )
    limits = [mink.ConfigurationLimit(model)]
    rng = np.random.default_rng(SEED)
    rows = []
    started = time.perf_counter()

    print(f"=== Multi-start offline IK: {len(failed)} failed pre-grasp pose(s) ===")
    print("Starts per pose: home +", N_RANDOM_STARTS, "random joint configurations")
    print("This does not test dynamics, contacts, collisions, or path feasibility.\n")

    for record in failed:
        ep = int(record["episode"])
        cube = np.asarray(record["initial_cube_pose_xyz_m"], dtype=float)
        yaw = float(record["initial_cube_yaw_rad"])
        pregrasp_object_center = cube + np.array([0.0, 0.0, 0.12])
        hand_rot = R.from_euler(
            "xyz", [180.0, 0.0, np.degrees(yaw)], degrees=True
        ).as_matrix()
        hand_pos = pregrasp_object_center + hand_rot @ np.array([0.0, 0.0, -0.105])
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(hand_rot), hand_pos
        )
        hand_task.set_target(target)

        starts = [("home", base_qpos.copy())]
        for j in range(N_RANDOM_STARTS):
            qpos = base_qpos.copy()
            qpos[arm_qpos] = rng.uniform(arm_ranges[:, 0], arm_ranges[:, 1])
            starts.append((f"random_{j+1}", qpos))

        print(f"Episode {ep}: cube=({cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}), yaw={yaw:+.3f} rad")
        any_success = False
        for start_name, qpos0 in starts:
            configuration = mink.Configuration(model)
            configuration.update(qpos0)
            ok = False
            pos_err = float("inf")
            ori_err = float("inf")
            iterations = 0
            for i in range(MAX_ITERS):
                iterations = i + 1
                velocity = mink.solve_ik(
                    configuration=configuration,
                    tasks=[hand_task],
                    dt=DT,
                    solver="daqp",
                    damping=1e-1,
                    limits=limits,
                )
                configuration.integrate_inplace(velocity, DT)
                err = hand_task.compute_error(configuration)
                pos_err = float(np.linalg.norm(err[:3]))
                ori_err = float(np.linalg.norm(err[3:]))
                if pos_err < POSITION_TOL and ori_err < ORIENTATION_TOL:
                    ok = True
                    break

            any_success = any_success or ok
            rows.append({
                "episode": ep,
                "initial_cube_x_m": float(cube[0]),
                "initial_cube_y_m": float(cube[1]),
                "initial_cube_z_m": float(cube[2]),
                "initial_cube_yaw_rad": yaw,
                "start_configuration": start_name,
                "ik_reached": ok,
                "iterations": iterations,
                "position_error_m": pos_err,
                "orientation_error_rad": ori_err,
            })
            print(
                f"  {start_name:8s}: {'REACHED' if ok else 'no convergence':14s} "
                f"pos={pos_err:.4f} m ori={ori_err:.4f} rad iters={iterations}"
            )
        print("  Summary:", "at least one static IK solution found" if any_success
              else "no static solution found from tested starts", "\n")

    RESULTS.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_poses = len(failed)
    any_count = sum(
        any(row["ik_reached"] for row in rows if row["episode"] == ep)
        for ep in {int(r["episode"]) for r in failed}
    )
    home_count = sum(row["ik_reached"] for row in rows if row["start_configuration"] == "home")
    text = (
        "Pre-grasp multi-start offline IK diagnostic\n"
        f"Source benchmark: {SOURCE_JSON.name}\n"
        f"Failed pre-grasp poses tested: {total_poses}\n"
        f"Poses reached from at least one tested start: {any_count}/{total_poses}\n"
        f"Poses reached from home start: {home_count}/{total_poses}\n"
        f"Random starts per pose: {N_RANDOM_STARTS}\n"
        f"Iterations per attempt: {MAX_ITERS}\n"
        "Interpretation: success demonstrates a kinematic IK solution from a tested start only; it does not prove collision-free path feasibility or successful physical execution.\n"
        f"Elapsed wall time: {time.perf_counter() - started:.2f} s\n"
    )
    OUT_TXT.write_text(text)
    print(text)
    print(f"[SAVED] {OUT_CSV}")
    print(f"[SAVED] {OUT_TXT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
