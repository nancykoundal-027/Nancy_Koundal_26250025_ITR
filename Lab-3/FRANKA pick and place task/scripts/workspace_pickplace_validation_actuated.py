"""Actuator/dynamics-aligned validation of the benchmark's pre-grasp and grasp IK.

Runs only the first two waypoints for the same seeded cube poses as the Mink
benchmark. It mirrors the benchmark's home reset, actuator setup, settling steps,
Cartesian ramp, DAQP/Mink controller, q-command update, and MuJoCo stepping.
It does not close the gripper, validate lifting, or test tray placement.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mink
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT_DIR / "robot_descriptions" / "franka_emika_panda" / "pick_place_scene.xml"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
DT = 0.002
MAX_STEPS_PER_MOVE = 3000
MAX_ARM_SPEED = 0.45
CARTESIAN_RAMP_STEPS = 500
POSITION_TOL = 0.010
ORIENTATION_TOL = 0.08
GRIPPER_OPEN = 255.0
HAND_OFFSET = np.array([0.0, 0.0, -0.105])
X_RANGE = (0.35, 0.55)
Y_RANGE = (-0.20, 0.20)
YAW_RANGE = (-np.pi, np.pi)


def named_id(model, object_type, name):
    idx = mujoco.mj_name2id(model, object_type, name)
    if idx < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=25,
                        help="Number of seeded episodes to validate (default: 25).")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_MOVE,
                        help="Maximum steps per waypoint (default: 3000).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base episode seed (default: 0).")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.max_steps < CARTESIAN_RAMP_STEPS + 1:
        parser.error("--max-steps must be greater than the Cartesian ramp length")
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
    arm_actuator_ids = [named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]
    gripper_actuator_id = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    hand_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    table_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    cube_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])
    cube_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    cube_half_z = float(model.geom_size[cube_geom_id, 2])
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    # Establish table height using the same reset strategy as the benchmark.
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[arm_qpos_ids] = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    for finger_name in ("finger_joint1", "finger_joint2"):
        jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
        data.qpos[model.jnt_qposadr[jid]] = 0.04
    mujoco.mj_forward(model, data)
    table_surface_z = float(data.xpos[table_id, 2] + model.geom_size[table_geom_id, 2])

    configuration = mink.Configuration(model)
    hand_task = mink.FrameTask(
        frame_name="hand", frame_type="body", position_cost=1.0,
        orientation_cost=1.0, lm_damping=1.0,
    )
    limits = [mink.ConfigurationLimit(model)]

    def set_arm_controls(q_command):
        for k, actuator_id in enumerate(arm_actuator_ids):
            lo, hi = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = float(np.clip(q_command[k], lo, hi))

    def hand_target_components(object_center, yaw):
        rotation = R.from_euler("xyz", [180.0, 0.0, np.degrees(yaw)], degrees=True).as_matrix()
        hand_position = np.asarray(object_center, dtype=float) + rotation @ HAND_OFFSET
        return rotation, hand_position

    def reset_episode(episode_seed):
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        else:
            mujoco.mj_resetData(model, data)
            data.qpos[arm_qpos_ids] = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

        for finger_name in ("finger_joint1", "finger_joint2"):
            jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            data.qpos[model.jnt_qposadr[jid]] = 0.04
        mujoco.mj_forward(model, data)
        cube_center_z = table_surface_z + cube_half_z

        rng = np.random.default_rng(episode_seed)
        cube_x = float(rng.uniform(*X_RANGE))
        cube_y = float(rng.uniform(*Y_RANGE))
        cube_yaw = float(rng.uniform(*YAW_RANGE))
        cube_quat_xyzw = R.from_euler("z", cube_yaw).as_quat()
        data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = [
            cube_x, cube_y, cube_center_z,
            cube_quat_xyzw[3], cube_quat_xyzw[0], cube_quat_xyzw[1], cube_quat_xyzw[2],
        ]
        data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

        # Match the benchmark's position-actuator initialization and settle.
        for actuator_id, qpos_id in zip(arm_actuator_ids, arm_qpos_ids):
            data.ctrl[actuator_id] = float(data.qpos[qpos_id])
        data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
        for _ in range(100):
            mujoco.mj_step(model, data)
        configuration.update(data.qpos)
        return np.array([cube_x, cube_y, cube_center_z], dtype=float), cube_yaw

    def reach_target(label, object_center, yaw):
        target_rotation, target_hand_position = hand_target_components(object_center, yaw)
        start_hand_position = data.xpos[hand_id].copy()
        start_rotation_matrix = data.xmat[hand_id].reshape(3, 3).copy()
        rotation_path = Slerp(
            [0.0, 1.0],
            R.from_matrix(np.stack([start_rotation_matrix, target_rotation])),
        )
        q_command = data.qpos[arm_qpos_ids].copy()
        data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
        last_pos_err = float("inf")
        last_ori_err = float("inf")

        for step in range(args.max_steps):
            u = min(step / float(CARTESIAN_RAMP_STEPS), 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            ramp_position = start_hand_position + blend * (target_hand_position - start_hand_position)
            ramp_rotation = rotation_path([blend])[0].as_matrix()
            hand_task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(ramp_rotation), ramp_position
            ))
            configuration.update(data.qpos)
            try:
                velocity = mink.solve_ik(
                    configuration=configuration,
                    tasks=[hand_task],
                    dt=model.opt.timestep,
                    solver="daqp",
                    damping=1e-1,
                    limits=limits,
                )
            except Exception as exc:
                return False, step + 1, last_pos_err, last_ori_err, f"Mink solver error: {exc}"

            arm_velocity = np.clip(velocity[arm_dof_ids], -MAX_ARM_SPEED, MAX_ARM_SPEED)
            q_command += arm_velocity * model.opt.timestep
            set_arm_controls(q_command)
            data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
            mujoco.mj_step(model, data)

            if step >= CARTESIAN_RAMP_STEPS and step % 10 == 0:
                hand_task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(target_rotation), target_hand_position
                ))
                configuration.update(data.qpos)
                error = hand_task.compute_error(configuration)
                last_pos_err = float(np.linalg.norm(error[:3]))
                last_ori_err = float(np.linalg.norm(error[3:]))
                if last_pos_err < POSITION_TOL and last_ori_err < ORIENTATION_TOL:
                    return True, step + 1, last_pos_err, last_ori_err, "reached"

        return False, args.max_steps, last_pos_err, last_ori_err, "IK did not converge"

    rows = []
    started = time.perf_counter()
    print(f"=== Actuator-aligned pick-and-place IK validation: {args.episodes} poses ===")
    print(f"Scene: {SCENE_PATH}")
    print("Uses benchmark reset, actuator commands, 100 settling steps, and MuJoCo stepping.")
    print("Tests pre-grasp and grasp approach only; no gripper closure or tray placement.")

    for ep in range(args.episodes):
        seed = args.seed + ep
        cube_center, cube_yaw = reset_episode(seed)
        pre_target = cube_center + np.array([0.0, 0.0, 0.12])
        pre_ok, pre_steps, pre_pos, pre_ori, pre_reason = reach_target("Pre-grasp", pre_target, cube_yaw)
        grasp_ok = False
        grasp_steps = 0
        grasp_pos = float("nan")
        grasp_ori = float("nan")
        grasp_reason = "not attempted: pre-grasp failed"
        if pre_ok:
            grasp_ok, grasp_steps, grasp_pos, grasp_ori, grasp_reason = reach_target(
                "Grasp approach", cube_center, cube_yaw
            )

        row = {
            "episode": ep + 1,
            "seed": seed,
            "cube_x_m": float(cube_center[0]),
            "cube_y_m": float(cube_center[1]),
            "cube_z_m": float(cube_center[2]),
            "cube_yaw_rad": float(cube_yaw),
            "pregrasp_reached": bool(pre_ok),
            "pregrasp_steps": int(pre_steps),
            "pregrasp_position_error_m": float(pre_pos),
            "pregrasp_orientation_error_rad": float(pre_ori),
            "pregrasp_reason": pre_reason,
            "grasp_approach_reached": bool(grasp_ok) if pre_ok else "",
            "grasp_approach_steps": int(grasp_steps) if pre_ok else "",
            "grasp_position_error_m": float(grasp_pos) if pre_ok else "",
            "grasp_orientation_error_rad": float(grasp_ori) if pre_ok else "",
            "grasp_reason": grasp_reason,
            "both_waypoints_reached": bool(pre_ok and grasp_ok),
        }
        rows.append(row)
        state = "PASS" if row["both_waypoints_reached"] else "FAIL"
        print(
            f"Episode {ep+1:02d} | {state:4s} | x={cube_center[0]:.3f}, y={cube_center[1]:.3f}, "
            f"yaw={cube_yaw:+.3f} | pregrasp={'OK' if pre_ok else 'FAIL'} "
            f"({pre_pos:.4f} m, {pre_ori:.4f} rad) | "
            f"grasp={'OK' if grasp_ok else ('FAIL' if pre_ok else 'SKIP')}"
        )

    suffix = f"{args.episodes}_episodes_seed{args.seed}"
    csv_path = RESULTS_DIR / f"pickplace_workspace_validation_actuated_{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    passed_pre = sum(bool(r["pregrasp_reached"]) for r in rows)
    attempted_grasp = [r for r in rows if r["pregrasp_reached"]]
    passed_grasp = sum(bool(r["grasp_approach_reached"]) for r in attempted_grasp)
    passed_both = sum(bool(r["both_waypoints_reached"]) for r in rows)
    grasp_pct = 100 * passed_grasp / len(attempted_grasp) if attempted_grasp else 0.0
    summary_path = RESULTS_DIR / f"pickplace_workspace_validation_actuated_{suffix}_summary.txt"
    summary = (
        "Actuator-aligned pick-and-place IK validation\n"
        f"Scene: {SCENE_PATH}\n"
        f"Seed base: {args.seed}; episodes: {len(rows)}\n"
        f"Pre-grasp reached: {passed_pre}/{len(rows)} ({100*passed_pre/len(rows):.1f}%)\n"
        f"Grasp approach reached after pre-grasp: {passed_grasp}/{len(attempted_grasp)} "
        f"({grasp_pct:.1f}% of attempted cases)\n"
        f"Both waypoints reached: {passed_both}/{len(rows)} ({100*passed_both/len(rows):.1f}%)\n"
        f"Elapsed wall time: {time.perf_counter()-started:.2f} s\n\n"
        "Limitations: this validates only the first two waypoints with actuator dynamics. "
        "It does not test grasp contact, cube lift, transit, tray placement, or full episode success.\n"
    )
    summary_path.write_text(summary)

    fig, ax = plt.subplots(figsize=(8, 6))
    pass_x = [r["cube_x_m"] for r in rows if r["both_waypoints_reached"]]
    pass_y = [r["cube_y_m"] for r in rows if r["both_waypoints_reached"]]
    fail_x = [r["cube_x_m"] for r in rows if not r["both_waypoints_reached"]]
    fail_y = [r["cube_y_m"] for r in rows if not r["both_waypoints_reached"]]
    if pass_x:
        ax.scatter(pass_x, pass_y, marker="o", label="Both waypoints reached")
    if fail_x:
        ax.scatter(fail_x, fail_y, marker="x", label="IK failure")
    ax.set_xlabel("Cube X (m)")
    ax.set_ylabel("Cube Y (m)")
    ax.set_title("Actuator-aligned pre-grasp / grasp IK validation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plot_path = PLOTS_DIR / f"pickplace_workspace_validation_actuated_{suffix}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("\n" + summary)
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
