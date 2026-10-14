import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mujoco
import mink

# 1. Directory and Path Context Setup
ROOT_DIR = os.getcwd()
SCENE_DIR = os.path.abspath("robot_descriptions")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the correct Franka Panda pick-and-place scene
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_DIR = os.path.join(
    ROOT_DIR, "robot_descriptions", "franka_emika_panda"
)

model = mujoco.MjModel.from_xml_path(
    os.path.join(SCENE_DIR, "pick_place_scene.xml")
)
data = mujoco.MjData(model)

# 2. Mink Task Setup
configuration = mink.Configuration(model)

ee_task = mink.FrameTask(
    frame_name="hand",
    frame_type="body",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1.0,
)
posture_task = mink.PostureTask(model, cost=1e-3)
limits = [mink.ConfigurationLimit(model)]
tasks = [ee_task]

# Bounds
X_RANGE = [0.35, 0.55]
Y_RANGE = [-0.20, 0.20]
YAW_RANGE = [-np.pi, np.pi]
TABLE_Z = 0.30


def get_hand_se3(target_pos, yaw):
    """Computes target SE3 pose for hand body."""
    rot_matrix = R.from_euler("xyz", [180, 0, np.degrees(yaw)], degrees=True).as_matrix()
    hand_pos = target_pos + rot_matrix @ np.array([0, 0, -0.05])
    so3 = mink.SO3.from_matrix(rot_matrix)
    return mink.SE3.from_rotation_and_translation(so3, hand_pos)


def run_episode(episode_id):
    np.random.seed(episode_id)
    mujoco.mj_resetData(model, data)

    # Home joint configuration
    home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
    data.qpos[:len(home_qpos)] = home_qpos
    mujoco.mj_forward(model, data)

    # Initialize Mink Configuration state and Posture Target
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

    # Randomize Cube Pose
    cube_x = np.random.uniform(*X_RANGE)
    cube_y = np.random.uniform(*Y_RANGE)
    cube_yaw = np.random.uniform(*YAW_RANGE)

    cube_id = model.body("cube").id
    tray_id = model.body("tray").id

    # Place cube on table
    cube_joint_id = model.joint("cube_free").id
    cube_qadr = int(model.jnt_qposadr[cube_joint_id])

    # Set the cube's actual free-joint position and randomized yaw.
    cube_quat_xyzw = R.from_euler("z", cube_yaw).as_quat()
    data.qpos[cube_qadr:cube_qadr + 3] = [
        cube_x, cube_y, TABLE_Z + 0.025
    ]
    # MuJoCo quaternion order is w, x, y, z.
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [
        cube_quat_xyzw[3],
        cube_quat_xyzw[0],
        cube_quat_xyzw[1],
        cube_quat_xyzw[2],
    ]
    mujoco.mj_forward(model, data)

    cube_pos = np.array([cube_x, cube_y, TABLE_Z + 0.025])
    tray_pos = data.xpos[tray_id].copy()

    # Sequence of target waypoints
    phases = [
        ("Pre-Grasp", cube_pos + np.array([0, 0, 0.12]), cube_yaw, False),
        ("Grasp", cube_pos + np.array([0, 0, 0.05]), cube_yaw, True),
        ("Lift", cube_pos + np.array([0, 0, 0.20]), cube_yaw, True),
        ("Transit", tray_pos + np.array([0, 0, 0.18]), 0.0, True),
        ("Place", tray_pos + np.array([0, 0, 0.10]), 0.0, False),
    ]

    total_iters = 0
    start_time = time.time()
    min_clearance = float("inf")

    for phase_name, target_pos, yaw, close_gripper in phases:
        target_se3 = get_hand_se3(target_pos, yaw)
        ee_task.set_target(target_se3)

        phase_converged = False
        # Increased iteration budget to 300 steps per phase
        for step in range(300):
            configuration.update(data.qpos)

            try:
                vel = mink.solve_ik(
                    configuration=configuration,
                    tasks=tasks,
                    dt=model.opt.timestep,
                    solver="daqp",
                    damping=1e-1,
                    limits=limits,
                )
            except Exception as e:
                return {
                    "episode": episode_id + 1,
                    "success": False,
                    "reason": f"Mink Solver Error: {str(e)}",
                    "time": time.time() - start_time,
                    "iters": total_iters,
                    "clearance": min_clearance if min_clearance != float("inf") else 0.0,
                }

            configuration.integrate_inplace(vel, model.opt.timestep)
            data.qpos[:7] = configuration.q[:7]

            # Actuate gripper fingers
            if model.nu >= 8:
                data.ctrl[7] = 0.0 if close_gripper else float(model.actuator_ctrlrange[7, 1])

            mujoco.mj_step(model, data)
            total_iters += 1

            # Monitor contacts
            for i in range(data.ncon):
                contact = data.contact[i]
                if contact.dist < min_clearance:
                    min_clearance = contact.dist

            # Check target convergence
            err = ee_task.compute_error(configuration)
            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:])
            if step == 299:
                print(
                    f"DEBUG {phase_name}: "
                    f"position error={pos_err:.6f}, "
                    f"orientation error={ori_err:.6f}, "
                    f"target={target_pos}"
               )
            if pos_err < 1e-2  and ori_err < 1e-1:
                phase_converged = True
                break

        if not phase_converged:
            return {
                "episode": episode_id + 1,
                "success": False,
                "reason": f"IK divergence in phase {phase_name}",
                "time": time.time() - start_time,
                "iters": total_iters,
                "clearance": float(min_clearance if min_clearance != float("inf") else 0.0),
            }

    solve_time = time.time() - start_time
    return {
        "episode": episode_id + 1,
        "success": True,
        "reason": "None",
        "time": solve_time,
        "iters": total_iters,
        "clearance": float(min_clearance if min_clearance != float("inf") else 0.0),
    }


if __name__ == "__main__":
    print("=== MINK IK DIAGNOSTIC TEST (1 EPISODE) ===")
    print(f"{'Episode':<10}{'Success':<10}{'Time (s)':<12}{'IK Iters':<12}{'Failure Reason':<30}")
    print("-" * 74)

    logs = []
    for ep in range(1):
        log = run_episode(ep)
        logs.append(log)
        print(f"{log['episode']:<10}{str(log['success']):<10}{log['time']:<12.3f}{log['iters']:<12}{log['reason']:<30}")

    # Save output JSON to results folder
    json_path = os.path.join(RESULTS_DIR, "mink_results_smoke.json")
    with open(json_path, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"\n[SAVED] Episode logs saved to '{json_path}'")

    # Plot metrics to plots folder
    successes = [l["success"] for l in logs]
    times = [l["time"] for l in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(["Success", "Failure"], [sum(successes), len(logs) - sum(successes)], color=['green', 'red'])
    ax1.set_title(f"Mink IK Success Rate ({sum(successes)/len(logs)*100:.1f}%)", fontweight="bold")
    ax1.set_ylabel("Episodes")

    ax2.plot(range(1, len(times) + 1), times, "o-", color='blue', label="Solve Time (s)")
    ax2.set_title("Solve Time per Episode", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True)

    plot_path = os.path.join(PLOTS_DIR, "mink_performance_smoke.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[SAVED] Performance plot saved to '{plot_path}'")
