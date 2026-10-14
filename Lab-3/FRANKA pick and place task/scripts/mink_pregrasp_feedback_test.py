"""Randomized Franka Panda pick-and-place benchmark using the working Mink controller.

Uses the same smooth Cartesian target ramp, actuator-position commands, arm speed
limit, hand/object offset, and waypoint sequence as mink_pick_place_demo_cartesian_smooth.py.

Default behavior is ONE diagnostic episode. Run --episodes 25 only after checking
that the diagnostic succeeds. Output files are deliberately separate from the old
mink_results.json and mink_performance.png files.
"""
from __future__ import annotations

import argparse
import json
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
GRIPPER_CLOSED = 0.0

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
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (default: 1 diagnostic episode).")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed. Episode i uses seed+i.")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene XML not found: {SCENE_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    arm_joint_ids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids], dtype=int)
    arm_dof_ids = np.array([model.jnt_dofadr[j] for j in arm_joint_ids], dtype=int)
    arm_actuator_ids = [named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]
    gripper_actuator_id = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    hand_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    tray_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    table_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    cube_joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])
    cube_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    tray_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "tray_base")
    cube_half_z = float(model.geom_size[cube_geom_id, 2])

    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    configuration = mink.Configuration(model)
    hand_task = mink.FrameTask(
        frame_name="hand", frame_type="body", position_cost=1.0,
        orientation_cost=1.0, lm_damping=1.0,
    )
    limits = [mink.ConfigurationLimit(model)]

    def hand_target_components(object_center, yaw):
        rotation = R.from_euler("xyz", [180.0, 0.0, np.degrees(yaw)], degrees=True).as_matrix()
        # Match the verified working demo's 10.5 cm hand/contact-centre offset.
        hand_position = np.asarray(object_center, dtype=float) + rotation @ np.array([0.0, 0.0, -0.105])
        return rotation, hand_position

    def set_arm_controls(q_command):
        for k, actuator_id in enumerate(arm_actuator_ids):
            lo, hi = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = float(np.clip(q_command[k], lo, hi))

    contact_min = [float("inf")]
    contact_pairs = set()

    def record_contacts():
        for i in range(data.ncon):
            c = data.contact[i]
            contact_min[0] = min(contact_min[0], float(c.dist))
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or f"geom_{int(c.geom1)}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or f"geom_{int(c.geom2)}"
            contact_pairs.add(" <-> ".join(sorted((n1, n2))))

    def reset_episode(episode_seed):
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        else:
            mujoco.mj_resetData(model, data)
            fallback = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
            data.qpos[arm_qpos_ids] = fallback

        # Open the fingers at reset and settle at the home posture.
        for finger_name in ("finger_joint1", "finger_joint2"):
            jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            data.qpos[model.jnt_qposadr[jid]] = 0.04
        mujoco.mj_forward(model, data)

        table_surface_z = float(data.xpos[table_id, 2] + model.geom_size[table_geom_id, 2])
        tray_surface_z = float(data.xpos[tray_id, 2] + model.geom_size[tray_geom_id, 2])
        tray_body_pos = data.xpos[tray_id].copy()

        rng = np.random.default_rng(episode_seed)
        cube_x = float(rng.uniform(*X_RANGE))
        cube_y = float(rng.uniform(*Y_RANGE))
        cube_yaw = float(rng.uniform(*YAW_RANGE))
        cube_center_z = table_surface_z + cube_half_z
        cube_quat_xyzw = R.from_euler("z", cube_yaw).as_quat()
        data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = np.array([
            cube_x, cube_y, cube_center_z,
            cube_quat_xyzw[3], cube_quat_xyzw[0], cube_quat_xyzw[1], cube_quat_xyzw[2],
        ])  # MuJoCo free-joint quaternion order is w, x, y, z.
        data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

        for actuator_id, qpos_id in zip(arm_actuator_ids, arm_qpos_ids):
            data.ctrl[actuator_id] = float(data.qpos[qpos_id])
        data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
        for _ in range(100):
            mujoco.mj_step(model, data)
        configuration.update(data.qpos)
        contact_min[0] = float("inf")
        contact_pairs.clear()
        return {
            "cube_start": np.array([cube_x, cube_y, cube_center_z]),
            "cube_yaw": cube_yaw,
            "table_surface_z": table_surface_z,
            "tray_surface_z": tray_surface_z,
            "tray_body_pos": tray_body_pos,
        }

    def move_to_object_center(label, object_center, yaw, close_gripper, metrics):
        target_rotation, target_hand_position = hand_target_components(object_center, yaw)
        start_hand_position = data.xpos[hand_id].copy()
        start_rotation_matrix = data.xmat[hand_id].reshape(3, 3).copy()
        rotation_path = Slerp(
            [0.0, 1.0],
            R.from_matrix(np.stack([start_rotation_matrix, target_rotation])),
        )
        q_command = data.qpos[arm_qpos_ids].copy()
        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if close_gripper else GRIPPER_OPEN
        last_pos_error = float("inf")
        last_ori_error = float("inf")

        for step in range(MAX_STEPS_PER_MOVE):
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
                metrics["phase_iterations"][label] = step + 1
                metrics["phase_errors"][label] = {"position_m": last_pos_error, "orientation_rad": last_ori_error}
                return False, q_command, f"Mink solver error during {label}: {exc}"

            arm_velocity = np.clip(velocity[arm_dof_ids], -MAX_ARM_SPEED, MAX_ARM_SPEED)
            q_command = data.qpos[arm_qpos_ids].copy() + arm_velocity * model.opt.timestep
            set_arm_controls(q_command)
            data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if close_gripper else GRIPPER_OPEN
            mujoco.mj_step(model, data)
            metrics["ik_iterations"] += 1
            metrics["simulation_steps"] += 1
            record_contacts()

            # Diagnostic only: measure error against the final target during motion.
            if step % 250 == 0:
                hand_task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3.from_matrix(target_rotation),
                        target_hand_position,
                    )
                )
                configuration.update(data.qpos)
                trace_error = hand_task.compute_error(configuration)
                print(
                    f"[TRACE {label}] step={step} "
                    f"hand={np.round(data.xpos[hand_id], 4)} "
                    f"position_error={np.linalg.norm(trace_error[:3]):.4f} m "
                    f"orientation_error={np.linalg.norm(trace_error[3:]):.4f} rad "
                    f"actual_q={np.round(data.qpos[arm_qpos_ids], 3)} "
                    f"command_q={np.round(q_command, 3)}"
                )

            # Keep matching the working controller: check the final, non-ramping target
            # only after the ramp has completed, every 10 steps.
            if step >= CARTESIAN_RAMP_STEPS and step % 10 == 0:
                hand_task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(target_rotation), target_hand_position
                ))
                configuration.update(data.qpos)
                error = hand_task.compute_error(configuration)
                last_pos_error = float(np.linalg.norm(error[:3]))
                last_ori_error = float(np.linalg.norm(error[3:]))
                if last_pos_error < POSITION_TOL and last_ori_error < ORIENTATION_TOL:
                    metrics["phase_iterations"][label] = step + 1
                    metrics["phase_errors"][label] = {"position_m": last_pos_error, "orientation_rad": last_ori_error}
                    return True, q_command, ""

        metrics["phase_iterations"][label] = MAX_STEPS_PER_MOVE
        metrics["phase_errors"][label] = {"position_m": last_pos_error, "orientation_rad": last_ori_error}
        return False, q_command, f"IK did not converge during {label} (position={last_pos_error:.4f} m, orientation={last_ori_error:.4f} rad)"

    def hold_gripper(closed, steps, q_command, metrics):
        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        set_arm_controls(q_command)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            metrics["simulation_steps"] += 1
            record_contacts()

    def run_episode(episode_index):
        seed = args.seed + episode_index
        start_time = time.perf_counter()
        metrics = {
            "episode": episode_index + 1,
            "seed": seed,
            "success": False,
            "reason": "Not completed",
            "time_s": 0.0,
            "ik_iterations": 0,
            "simulation_steps": 0,
            "phase_iterations": {},
            "phase_errors": {},
            "min_contact_distance_m": None,
            "contact_pairs": [],
            "initial_cube_pose_xyz_m": None,
            "initial_cube_yaw_rad": None,
            "cube_lifted": False,
            "placed_in_tray": False,
            "final_cube_position_xyz_m": None,
        }
        try:
            pose = reset_episode(seed)
            cube_start = pose["cube_start"]
            cube_yaw = pose["cube_yaw"]
            tray_surface_z = pose["tray_surface_z"]
            tray_body_pos = pose["tray_body_pos"]
            metrics["initial_cube_pose_xyz_m"] = cube_start.tolist()
            metrics["initial_cube_yaw_rad"] = cube_yaw
            q_command = data.qpos[arm_qpos_ids].copy()

            tray_place_center = np.array([
                tray_body_pos[0], tray_body_pos[1], tray_surface_z + cube_half_z + 0.003
            ])
            tray_transit_center = tray_place_center + np.array([0.0, 0.0, 0.10])

            # Stage sequence mirrors the confirmed working single-episode demo.
            stages = [
                ("Pre-grasp", cube_start + np.array([0.0, 0.0, 0.12]), cube_yaw, False),
                ("Grasp approach", cube_start, cube_yaw, False),
            ]
            for label, center, yaw, closed in stages:
                ok, q_command, reason = move_to_object_center(label, center, yaw, closed, metrics)
                if not ok:
                    metrics["reason"] = reason
                    return finish(metrics, start_time)

            hold_gripper(True, 600, q_command, metrics)
            cube_z_before_lift = float(data.xpos[cube_id, 2])
            ok, q_command, reason = move_to_object_center(
                "Lift", cube_start + np.array([0.0, 0.0, 0.12]), cube_yaw, True, metrics
            )
            cube_z_after_lift = float(data.xpos[cube_id, 2])
            metrics["cube_z_before_lift_m"] = cube_z_before_lift
            metrics["cube_z_after_lift_m"] = cube_z_after_lift
            metrics["cube_lift_delta_m"] = cube_z_after_lift - cube_z_before_lift
            metrics["cube_lifted"] = cube_z_after_lift >= cube_z_before_lift + 0.015
            if not ok:
                metrics["reason"] = reason
                return finish(metrics, start_time)
            if not metrics["cube_lifted"]:
                metrics["reason"] = "Grasp check failed: cube did not rise by at least 0.015 m"
                return finish(metrics, start_time)

            for label, center, yaw, closed in [
                ("Transit above tray", tray_transit_center, cube_yaw, True),
                ("Place in tray", tray_place_center, cube_yaw, True),
            ]:
                ok, q_command, reason = move_to_object_center(label, center, yaw, closed, metrics)
                if not ok:
                    metrics["reason"] = reason
                    return finish(metrics, start_time)

            hold_gripper(False, 350, q_command, metrics)
            for _ in range(300):
                mujoco.mj_step(model, data)
                metrics["simulation_steps"] += 1
                record_contacts()
            final_cube = data.xpos[cube_id].copy()
            metrics["final_cube_position_xyz_m"] = final_cube.tolist()
            inside_xy = (
                abs(final_cube[0] - tray_body_pos[0]) < 0.075
                and abs(final_cube[1] - tray_body_pos[1]) < 0.075
            )
            rests_near_tray = (
                tray_surface_z - 0.01 <= final_cube[2]
                <= tray_surface_z + 2.0 * cube_half_z + 0.07
            )
            metrics["placed_in_tray"] = bool(inside_xy and rests_near_tray)
            if not metrics["placed_in_tray"]:
                metrics["reason"] = "Placement check failed: cube not confirmed inside tray at plausible height"
                return finish(metrics, start_time)

            ok, _q_command, reason = move_to_object_center(
                "Retreat", tray_place_center + np.array([0.0, 0.0, 0.12]), cube_yaw, False, metrics
            )
            if not ok:
                # Placement is the task success criterion; report retreat issue separately.
                metrics["retreat_warning"] = reason
            metrics["success"] = True
            metrics["reason"] = "Cube lifted and confirmed in tray"
            return finish(metrics, start_time)
        except Exception as exc:
            metrics["reason"] = f"Episode exception: {type(exc).__name__}: {exc}"
            return finish(metrics, start_time)

    def finish(metrics, start_time):
        metrics["time_s"] = round(time.perf_counter() - start_time, 4)
        metrics["min_contact_distance_m"] = None if contact_min[0] == float("inf") else float(contact_min[0])
        # These are observed active contact pairs. Signed contact distances can be
        # negative due to penetration; they are not a global free-space clearance.
        metrics["contact_pairs"] = sorted(contact_pairs)
        return metrics

    print(f"=== Mink randomized benchmark: {args.episodes} episode(s), seed base {args.seed} ===")
    print("Controller: same smooth Cartesian ramp + DAQP/Mink IK as the working demo")
    print(f"Scene: {SCENE_PATH}")
    logs = []
    for ep in range(args.episodes):
        result = run_episode(ep)
        logs.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"Episode {result['episode']:02d} | {status:<7} | time={result['time_s']:.2f}s | "
            f"IK iters={result['ik_iterations']} | reason={result['reason']}"
        )
        if result.get("phase_errors"):
            for phase, err in result["phase_errors"].items():
                print(f"  {phase}: pos={err['position_m']:.4f} m, ori={err['orientation_rad']:.4f} rad; steps={result['phase_iterations'].get(phase)}")
        if result.get("final_cube_position_xyz_m") is not None:
            print(f"  Final cube position: {np.round(result['final_cube_position_xyz_m'], 4)}")

    success_count = sum(bool(row["success"]) for row in logs)
    suffix = "diagnostic" if args.episodes == 1 else f"{args.episodes}_episodes"
    json_path = RESULTS_DIR / f"mink_results_pregrasp_feedback_test_{suffix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    plot_path = PLOTS_DIR / f"mink_performance_pregrasp_feedback_test_{suffix}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(["Success", "Failure"], [success_count, len(logs) - success_count])
    ax1.set_title(f"Mink pick-and-place success: {success_count}/{len(logs)}")
    ax1.set_ylabel("Episodes")
    times = [row["time_s"] for row in logs]
    ax2.plot(range(1, len(logs) + 1), times, "o-")
    ax2.set_title("Wall-clock time per episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Time (s)")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"\nSummary: {success_count}/{len(logs)} successes ({100*success_count/len(logs):.1f}%)")
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {plot_path}")
    if args.episodes == 1 and not logs[0]["success"]:
        print("Diagnostic did not succeed. Inspect its failure phase before scaling to 25 episodes.")


if __name__ == "__main__":
    main()
