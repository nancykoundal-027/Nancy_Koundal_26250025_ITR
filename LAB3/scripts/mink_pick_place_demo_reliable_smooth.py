#!/usr/bin/env python3
"""Franka Panda pick-and-place with very mild joint smoothing and grasp safety checks.

Run from the project root with:
    conda activate ros_env
    mjpython scripts/mink_pick_place_demo_reliable_smooth.py

This deliberately runs ONE fixed cube placement in the viewer. It does not run
benchmark episodes. The existing benchmark script is left untouched.
"""
from pathlib import Path
import time
import sys

import mujoco
import mujoco.viewer
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT_DIR / "robot_descriptions" / "franka_emika_panda" / "pick_place_scene.xml"
DT = 0.002
MAX_STEPS_PER_MOVE = 2600
MAX_ARM_SPEED = 0.45  # rad/s, keep the speed used by the working demo
MAX_ARM_ACCEL = 15.0  # rad/s^2, light acceleration limiting
VELOCITY_FILTER_ALPHA = 0.95  # very light filtering, close to the working controller
POSITION_TOL = 0.010  # m
ORIENTATION_TOL = 0.08  # rad
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def named_id(model, object_type, name):
    idx = mujoco.mj_name2id(model, object_type, name)
    if idx < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return idx


def main():
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene XML not found: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    # Resolve named robot joints and actuators instead of assuming arbitrary order.
    arm_joint_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 8)
    ]
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids], dtype=int)
    arm_dof_ids = np.array([model.jnt_dofadr[j] for j in arm_joint_ids], dtype=int)
    arm_actuator_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
        for i in range(1, 8)
    ]
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

    # Reset to the model's verified home keyframe when available.
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        home_arm = data.qpos[arm_qpos_ids].copy()
    else:
        mujoco.mj_resetData(model, data)
        home_arm = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        for qpos_id, value in zip(arm_qpos_ids, home_arm):
            data.qpos[qpos_id] = value

    # Open both fingers. In this model actuator8 uses a 0..255 control range.
    for finger_name in ("finger_joint1", "finger_joint2"):
        jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
        data.qpos[model.jnt_qposadr[jid]] = 0.04

    mujoco.mj_forward(model, data)
    table_surface_z = float(data.xpos[table_id, 2] + model.geom_size[table_geom_id, 2])
    tray_surface_z = float(data.xpos[tray_id, 2] + model.geom_size[tray_geom_id, 2])

    # Fixed, reproducible cube pose. A free joint is set through qpos, not xpos.
    cube_x, cube_y, cube_yaw = 0.45, 0.0, 0.0
    cube_center_z = table_surface_z + cube_half_z
    cube_quat_xyzw = R.from_euler("z", cube_yaw).as_quat()
    data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = np.array([
        cube_x, cube_y, cube_center_z,
        cube_quat_xyzw[3], cube_quat_xyzw[0], cube_quat_xyzw[1], cube_quat_xyzw[2],
    ])  # MuJoCo free-joint quaternion order is w, x, y, z.
    data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Initialize actuator commands at the current arm pose and gripper-open value.
    for actuator_id, qpos_id in zip(arm_actuator_ids, arm_qpos_ids):
        data.ctrl[actuator_id] = data.qpos[qpos_id]
    data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
    for _ in range(100):
        mujoco.mj_step(model, data)

    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    hand_task = mink.FrameTask(
        frame_name="hand",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    limits = [mink.ConfigurationLimit(model)]

    def make_hand_target(object_center, yaw):
        # Downward-facing hand. The 10.5 cm offset maps the desired finger-contact
        # point (near the cube centre) to the Panda hand-body frame.
        rotation = R.from_euler(
            "xyz", [180.0, 0.0, np.degrees(yaw)], degrees=True
        ).as_matrix()
        hand_position = np.asarray(object_center, dtype=float) + rotation @ np.array([0.0, 0.0, -0.105])
        return mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rotation), hand_position
        )

    def set_arm_controls(q_command):
        for k, actuator_id in enumerate(arm_actuator_ids):
            lo, hi = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = float(np.clip(q_command[k], lo, hi))

    def hold_gripper(closed, steps, viewer, q_command):
        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        set_arm_controls(q_command)
        for _ in range(steps):
            if not viewer.is_running():
                return
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(DT)

    def move_to_object_center(label, object_center, yaw, close_gripper, viewer):
        target_pose = make_hand_target(object_center, yaw)
        hand_task.set_target(target_pose)
        q_command = data.qpos[arm_qpos_ids].copy()
        # Initialize from the measured velocity so a waypoint change does not
        # introduce an artificial zero-velocity jump.
        previous_arm_velocity = np.clip(
            data.qvel[arm_dof_ids].copy(), -MAX_ARM_SPEED, MAX_ARM_SPEED
        )
        reached = False
        last_position_error = float("inf")
        last_orientation_error = float("inf")

        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if close_gripper else GRIPPER_OPEN
        print(f"\n[{label}] target object/contact centre: {np.asarray(object_center)}")

        for step in range(MAX_STEPS_PER_MOVE):
            if not viewer.is_running():
                return False, q_command

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
                print(f"[{label}] IK solver error: {exc}")
                return False, q_command

            # Keep the successful demo's target path and speed cap, while applying
            # only very mild low-pass filtering and a modest acceleration limit.
            # Do not write data.qpos directly; the MuJoCo position actuators move the arm.
            raw_velocity = np.clip(velocity[arm_dof_ids], -MAX_ARM_SPEED, MAX_ARM_SPEED)
            filtered_velocity = (
                VELOCITY_FILTER_ALPHA * raw_velocity
                + (1.0 - VELOCITY_FILTER_ALPHA) * previous_arm_velocity
            )
            max_dv = MAX_ARM_ACCEL * model.opt.timestep
            arm_velocity = previous_arm_velocity + np.clip(
                filtered_velocity - previous_arm_velocity, -max_dv, max_dv
            )
            previous_arm_velocity = arm_velocity.copy()
            q_command += arm_velocity * model.opt.timestep
            set_arm_controls(q_command)
            data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED if close_gripper else GRIPPER_OPEN

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(DT)

            if step % 10 == 0:
                configuration.update(data.qpos)
                error = hand_task.compute_error(configuration)
                last_position_error = float(np.linalg.norm(error[:3]))
                last_orientation_error = float(np.linalg.norm(error[3:]))
                if last_position_error < POSITION_TOL and last_orientation_error < ORIENTATION_TOL:
                    reached = True
                    break

        print(
            f"[{label}] reached={reached}; position error={last_position_error:.4f} m; "
            f"orientation error={last_orientation_error:.4f} rad"
        )
        print(f"[{label}] actual hand position: {data.xpos[hand_id].copy()}")
        return reached, q_command

    print("Scene:", SCENE_PATH)
    print(f"Table surface z={table_surface_z:.3f} m; tray surface z={tray_surface_z:.3f} m")
    print(f"Initial cube centre: {data.xpos[cube_id].copy()}")
    print("Starting one deterministic pick-and-place demo with very mild joint smoothing and a grasp check.")

    # Desired cube-centre waypoints. Tray centre is inside the tray's inner area.
    cube_start = np.array([cube_x, cube_y, cube_center_z])
    tray_body_pos = data.xpos[tray_id].copy()
    tray_place_center = np.array([
        tray_body_pos[0],
        tray_body_pos[1],
        tray_surface_z + cube_half_z + 0.003,
    ])
    tray_transit_center = tray_place_center + np.array([0.0, 0.0, 0.10])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        time.sleep(1.0)

        # 1. Approach with fingers open.
        ok, q_command = move_to_object_center(
            "Pre-grasp", cube_start + np.array([0.0, 0.0, 0.12]), cube_yaw, False, viewer
        )
        if not ok:
            print("Pre-grasp did not converge. Stopping before attempting to grasp.")
        else:
            # 2. Lower fingers to cube centre, still open.
            ok, q_command = move_to_object_center("Grasp approach", cube_start, cube_yaw, False, viewer)
            if not ok:
                print("Grasp approach did not converge. Stopping before closing the gripper.")
            else:
                # 3. Close gripper only after the approach is complete.
                print("Closing gripper (actuator8 control = 0).")
                hold_gripper(True, 300, viewer, q_command)
                finger_positions = []
                for finger_name in ("finger_joint1", "finger_joint2"):
                    jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
                    finger_positions.append(float(data.qpos[model.jnt_qposadr[jid]]))
                print("Finger joint positions after closing:", finger_positions)
                cube_z_before_lift = float(data.xpos[cube_id, 2])

                # 4. Lift, then carry above the tray while keeping fingers closed.
                ok, q_command = move_to_object_center(
                    "Lift", cube_start + np.array([0.0, 0.0, 0.12]), cube_yaw, True, viewer
                )
                cube_z_after_lift = float(data.xpos[cube_id, 2])
                print(
                    f"Cube z before lift={cube_z_before_lift:.3f} m; "
                    f"after lift={cube_z_after_lift:.3f} m"
                )
                cube_lifted = cube_z_after_lift >= cube_z_before_lift + 0.015
                if not cube_lifted:
                    print("GRASP CHECK FAILED: cube did not rise with the hand.")
                    print("Stopping here so the arm does not push/throw the ungrasped cube toward the tray.")
                    ok = False

                if ok and cube_lifted:
                    ok, q_command = move_to_object_center(
                        "Transit above tray", tray_transit_center, 0.0, True, viewer
                    )
                if ok:
                    ok, q_command = move_to_object_center(
                        "Place in tray", tray_place_center, 0.0, True, viewer
                    )
                if ok:
                    print("Opening gripper (actuator8 control = 255).")
                    hold_gripper(False, 350, viewer, q_command)
                    # Settle the released cube before evaluating its final location.
                    for _ in range(300):
                        if not viewer.is_running():
                            break
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(DT)
                    final_cube = data.xpos[cube_id].copy()
                    inside_xy = (
                        abs(final_cube[0] - tray_body_pos[0]) < 0.075
                        and abs(final_cube[1] - tray_body_pos[1]) < 0.075
                    )
                    rests_near_tray = (
                        tray_surface_z - 0.01 <= final_cube[2]
                        <= tray_surface_z + 2.0 * cube_half_z + 0.07
                    )
                    print("Final cube centre:", final_cube)
                    print("Cube inside tray footprint:", inside_xy)
                    print("Cube at plausible tray height:", rests_near_tray)
                    print("DEMO RESULT:", "PLACED IN TRAY (check viewer)" if inside_xy and rests_near_tray else "NOT CONFIRMED")
                    # Retreat with fingers open.
                    move_to_object_center(
                        "Retreat", tray_place_center + np.array([0.0, 0.0, 0.12]), 0.0, False, viewer
                    )

        print("Demo motion finished. Inspect the viewer; close its window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
