"""Validate pre-grasp and grasp-approach IK for the benchmark's seeded poses.

This is a kinematic reachability diagnostic. It does not simulate finger contact,
object lifting, collisions, tray placement, or full pick-and-place success.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mink
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT_DIR / "robot_descriptions" / "franka_emika_panda" / "pick_place_scene.xml"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
DT = 0.002
CARTESIAN_RAMP_STEPS = 500
MAX_STEPS_PER_MOVE = 3000
MAX_ARM_SPEED = 0.45
POSITION_TOL = 0.010
ORIENTATION_TOL = 0.08
HAND_OFFSET = np.array([0.0, 0.0, -0.105])
X_RANGE = (0.35, 0.55)
Y_RANGE = (-0.20, 0.20)
YAW_RANGE = (-np.pi, np.pi)


def named_id(model, object_type, name):
    idx = mujoco.mj_name2id(model, object_type, name)
    if idx < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return idx


def hand_target(object_center, yaw):
    rot = R.from_euler("xyz", [180.0, 0.0, np.degrees(yaw)], degrees=True).as_matrix()
    pos = np.asarray(object_center, dtype=float) + rot @ HAND_OFFSET
    target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos)
    return rot, pos, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", "--samples", type=int, default=25,
                        help="Number of seeded benchmark poses to test (default: 25).")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_MOVE,
                        help="Maximum kinematic IK iterations per waypoint.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene XML not found: {SCENE_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = DT
    data = mujoco.MjData(model)

    arm_joint_ids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids], dtype=int)
    arm_dof_ids = np.array([model.jnt_dofadr[j] for j in arm_joint_ids], dtype=int)
    joint_ranges = np.array([model.jnt_range[j] for j in arm_joint_ids], dtype=float)
    hand_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    table_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    cube_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])
    cube_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    cube_half_z = float(model.geom_size[cube_geom_id, 2])

    mujoco.mj_resetData(model, data)
    data.qpos[arm_qpos_ids] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    for finger_name in ("finger_joint1", "finger_joint2"):
        finger_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
        data.qpos[model.jnt_qposadr[finger_id]] = 0.04
    mujoco.mj_forward(model, data)
    table_surface_z = float(data.xpos[table_id, 2] + model.geom_size[table_geom_id, 2])

    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    hand_task = mink.FrameTask(
        frame_name="hand", frame_type="body", position_cost=1.0,
        orientation_cost=1.0, lm_damping=1.0,
    )
    limits = [mink.ConfigurationLimit(model)]

    def reset_pose(cube_x, cube_y, cube_yaw):
        mujoco.mj_resetData(model, data)
        data.qpos[arm_qpos_ids] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        for finger_name in ("finger_joint1", "finger_joint2"):
            finger_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            data.qpos[model.jnt_qposadr[finger_id]] = 0.04
        cube_center_z = table_surface_z + cube_half_z
        quat_xyzw = R.from_euler("z", cube_yaw).as_quat()
        data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = [
            cube_x, cube_y, cube_center_z,
            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2],
        ]
        data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        return np.array([cube_x, cube_y, cube_center_z], dtype=float)

    def reach_target(object_center, yaw, max_steps):
        target_rotation, target_hand_position, final_target = hand_target(object_center, yaw)
        start_hand_position = data.xpos[hand_id].copy()
        start_rotation = data.xmat[hand_id].reshape(3, 3).copy()
        rotation_path = Slerp([0.0, 1.0], R.from_matrix(np.stack([start_rotation, target_rotation])))
        q_command = data.qpos[arm_qpos_ids].copy()
        hand_task.set_target(final_target)
        last_pos_err = float("inf")
        last_ori_err = float("inf")

        for step in range(max_steps):
            configuration.update(data.qpos)
            u = min(step / float(CARTESIAN_RAMP_STEPS), 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            ramp_position = start_hand_position + blend * (target_hand_position - start_hand_position)
            ramp_rotation = rotation_path([blend])[0].as_matrix()
            hand_task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(ramp_rotation), ramp_position
            ))
            try:
                velocity = mink.solve_ik(
                    configuration=configuration,
                    tasks=[hand_task],
                    dt=DT,
                    solver="daqp",
                    damping=1e-1,
                    limits=limits,
                )
            except Exception as exc:
                return False, step + 1, last_pos_err, last_ori_err, f"solver error: {exc}"

            q_command += np.clip(velocity[arm_dof_ids], -MAX_ARM_SPEED, MAX_ARM_SPEED) * DT
            q_command = np.clip(q_command, joint_ranges[:, 0] + 1e-5, joint_ranges[:, 1] - 1e-5)
            data.qpos[arm_qpos_ids] = q_command
            mujoco.mj_forward(model, data)

            if step >= CARTESIAN_RAMP_STEPS and step % 10 == 0:
                hand_task.set_target(final_target)
                configuration.update(data.qpos)
                err = hand_task.compute_error(configuration)
                last_pos_err = float(np.linalg.norm(err[:3]))
                last_ori_err = float(np.linalg.norm(err[3:]))
                if last_pos_err < POSITION_TOL and last_ori_err < ORIENTATION_TOL:
                    return True, step + 1, last_pos_err, last_ori_err, "reached"

        return False, max_steps, last_pos_err, last_ori_err, "IK did not converge"

    rows = []
    started = time.time()
    print(f"=== Pick-and-place workspace IK validation: {args.episodes} seeded poses ===")
    print(f"Scene: {SCENE_PATH}")
    print(f"Table surface z={table_surface_z:.4f} m; cube half-height={cube_half_z:.4f} m")
    print("Tests full hand orientation + 0.105 m hand offset; no dynamics/contact simulation.")

    for ep in range(args.episodes):
        # Match the benchmark's seed-per-episode sampling exactly.
        rng = np.random.default_rng(ep)
        cube_x = float(rng.uniform(*X_RANGE))
        cube_y = float(rng.uniform(*Y_RANGE))
        cube_yaw = float(rng.uniform(*YAW_RANGE))
        cube_center = reset_pose(cube_x, cube_y, cube_yaw)
        pregrasp_target = cube_center + np.array([0.0, 0.0, 0.12])
        pre_ok, pre_steps, pre_pos, pre_ori, pre_reason = reach_target(
            pregrasp_target, cube_yaw, args.max_steps
        )

        grasp_ok = False
        grasp_steps = 0
        grasp_pos = float("nan")
        grasp_ori = float("nan")
        grasp_reason = "not attempted: pre-grasp failed"
        if pre_ok:
            grasp_ok, grasp_steps, grasp_pos, grasp_ori, grasp_reason = reach_target(
                cube_center, cube_yaw, args.max_steps
            )

        row = {
            "episode": ep + 1,
            "cube_x_m": cube_x,
            "cube_y_m": cube_y,
            "cube_z_m": float(cube_center[2]),
            "cube_yaw_rad": cube_yaw,
            "pregrasp_reached": pre_ok,
            "pregrasp_steps": pre_steps,
            "pregrasp_position_error_m": pre_pos,
            "pregrasp_orientation_error_rad": pre_ori,
            "pregrasp_reason": pre_reason,
            "grasp_approach_reached": grasp_ok if pre_ok else "",
            "grasp_approach_steps": grasp_steps if pre_ok else "",
            "grasp_position_error_m": grasp_pos if pre_ok else "",
            "grasp_orientation_error_rad": grasp_ori if pre_ok else "",
            "grasp_reason": grasp_reason,
            "both_waypoints_reached": bool(pre_ok and grasp_ok),
        }
        rows.append(row)
        state = "PASS" if row["both_waypoints_reached"] else "FAIL"
        print(
            f"Episode {ep + 1:02d} | {state:4s} | x={cube_x:.3f}, y={cube_y:.3f}, "
            f"yaw={cube_yaw:+.3f} | pregrasp={'OK' if pre_ok else 'FAIL'} "
            f"({pre_pos:.4f} m, {pre_ori:.4f} rad) | "
            f"grasp={'OK' if grasp_ok else ('FAIL' if pre_ok else 'SKIP')}"
        )

    csv_path = RESULTS_DIR / "pickplace_workspace_validation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    passed_pre = sum(bool(r["pregrasp_reached"]) for r in rows)
    attempted_grasp = [r for r in rows if r["pregrasp_reached"]]
    passed_grasp = sum(bool(r["grasp_approach_reached"]) for r in attempted_grasp)
    passed_both = sum(bool(r["both_waypoints_reached"]) for r in rows)
    grasp_pct = (100 * passed_grasp / len(attempted_grasp)) if attempted_grasp else 0.0
    summary_path = RESULTS_DIR / "pickplace_workspace_validation_summary.txt"
    summary = (
        f"Pick-and-place workspace IK validation\n"
        f"Scene: {SCENE_PATH}\n"
        f"Tested seeded poses: {len(rows)}\n"
        f"Pre-grasp reachable: {passed_pre}/{len(rows)} ({100*passed_pre/len(rows):.1f}%)\n"
        f"Grasp approach reachable after pre-grasp: {passed_grasp}/{len(attempted_grasp)} "
        f"({grasp_pct:.1f}% of attempted cases)\n"
        f"Both waypoints reachable: {passed_both}/{len(rows)} ({100*passed_both/len(rows):.1f}%)\n"
        f"Elapsed wall time: {time.time()-started:.2f} s\n\n"
        f"Important limitation: this is a kinematic IK reachability diagnostic, not a physical pick-and-place test.\n"
        f"It does not simulate gripper contact, object lift, collision clearance, transit to tray, or final placement.\n"
    )
    summary_path.write_text(summary)

    fig, ax = plt.subplots(figsize=(8, 6))
    fail_x = [r["cube_x_m"] for r in rows if not r["both_waypoints_reached"]]
    fail_y = [r["cube_y_m"] for r in rows if not r["both_waypoints_reached"]]
    pass_x = [r["cube_x_m"] for r in rows if r["both_waypoints_reached"]]
    pass_y = [r["cube_y_m"] for r in rows if r["both_waypoints_reached"]]
    if pass_x:
        ax.scatter(pass_x, pass_y, marker="o", label="Both waypoints reached")
    if fail_x:
        ax.scatter(fail_x, fail_y, marker="x", label="IK failure")
    ax.set_xlabel("Cube X (m)")
    ax.set_ylabel("Cube Y (m)")
    ax.set_title("Pick-and-place waypoint IK validation (seeded poses)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plot_path = PLOTS_DIR / "pickplace_workspace_validation.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("\n" + summary)
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
