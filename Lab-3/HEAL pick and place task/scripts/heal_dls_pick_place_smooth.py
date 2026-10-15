from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------
# 1. LOAD THE TESTED FRONT-TABLE SCENE
# ------------------------------------------------------------
SCENE = Path("scenes/heal_pick_place_gripper_front_table_50g.xml")

# Re-create the lighter cube scene only if it does not exist.
if not SCENE.exists():
    source = Path("scenes/heal_pick_place_gripper_front_table.xml")
    if not source.exists():
        raise FileNotFoundError(f"Scene not found: {source}")

    tree = ET.parse(source)
    cube_xml = tree.find(".//worldbody/body[@name='cube']")
    if cube_xml is None:
        raise RuntimeError("Cube body not found in the source scene.")

    inertial = cube_xml.find("inertial")
    if inertial is None:
        raise RuntimeError("Cube inertial element not found.")

    mass = 0.05
    inertia = mass * 0.05**2 / 6.0
    inertial.set("mass", str(mass))
    inertial.set(
        "diaginertia",
        f"{inertia:.9g} {inertia:.9g} {inertia:.9g}"
    )

    for geom in cube_xml.findall("geom"):
        geom.set("friction", "2.0 0.01 0.001")

    for name in ("left_pad", "right_pad"):
        body = tree.find(f".//body[@name='{name}']")
        if body is not None:
            for geom in body.findall("geom"):
                geom.set("friction", "2.0 0.01 0.001")

    tree.write(SCENE, encoding="utf-8", xml_declaration=True)

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
dt = float(model.opt.timestep)

joint_names = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]
actuator_names = [
    "turret", "shoulder", "elbow",
    "wrist_1", "wrist_2", "wrist_3"
]

joint_ids = np.array([model.joint(n).id for n in joint_names])
actuator_ids = np.array([model.actuator(n).id for n in actuator_names])
qadr = model.jnt_qposadr[joint_ids]
dofs = model.jnt_dofadr[joint_ids]
joint_ranges = model.jnt_range[joint_ids]
torque_limits = model.actuator_ctrlrange[actuator_ids]

gripper_id = model.actuator("fingers_actuator").id
site_id = model.site("right_center").id
cube_id = model.body("cube").id
tray_id = model.body("tray").id

# Same tested tool orientation and front-table arrangement.
TOOL_ROTATION = np.diag([-1.0, 1.0, -1.0])

PREGRASP = np.array([0.45, 0.10, 0.45])
GRASP = np.array([0.45, 0.10, 0.36])

# Important: lift straight up first, then carry horizontally.
LIFT = np.array([0.45, 0.10, 0.48])
TRAY_HOVER = np.array([0.35, -0.15, 0.48])
TRAY_PLACE = np.array([0.35, -0.15, 0.375])

KP = 120.0
KD = 15.0
DAMPING = 0.05
ORIENTATION_WEIGHT = 0.3
POSITION_GAIN = 4.5
ORIENTATION_GAIN = 2.5
MAX_JOINT_SPEED = 0.65

state = {
    "q_ref": data.qpos[qadr].copy(),
    "gripper": 0.0,
}

mujoco.mj_forward(model, data)
cube_start = data.xpos[cube_id].copy()

print("\n=== HEAL DLS SMOOTH PICK-AND-PLACE ===")
print("Scene:", SCENE)
print("Cube initial position:", np.round(cube_start, 4))
print("Path: approach -> descend -> grip -> vertical lift")
print("Then horizontal carry at high clearance -> vertical place -> release")


# ------------------------------------------------------------
# 2. ONLINE DAMPED-LEAST-SQUARES CONTROLLER
# ------------------------------------------------------------
def dls_step(position_target, feedforward_velocity):
    """Compute DLS joint velocity and track it with torque-limited PD."""
    mujoco.mj_forward(model, data)

    current_position = data.site_xpos[site_id].copy()
    current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    position_error = position_target - current_position

    orientation_error = (
        Rotation.from_matrix(TOOL_ROTATION)
        * Rotation.from_matrix(current_rotation).inv()
    ).as_rotvec()

    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)

    # Weight the angular rows so the solver balances position and orientation.
    jacobian = np.vstack((
        jac_p[:, dofs],
        ORIENTATION_WEIGHT * jac_r[:, dofs]
    ))

    desired_twist = np.concatenate((
        feedforward_velocity + POSITION_GAIN * position_error,
        ORIENTATION_WEIGHT * ORIENTATION_GAIN * orientation_error
    ))

    # Damped least squares: qdot = J.T (J J.T + lambda^2 I)^-1 v
    matrix = (
        jacobian @ jacobian.T
        + DAMPING**2 * np.eye(6)
    )
    qdot = jacobian.T @ np.linalg.solve(matrix, desired_twist)
    qdot = np.clip(qdot, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)

    q_ref = state["q_ref"] + qdot * dt
    q_ref = np.clip(
        q_ref,
        joint_ranges[:, 0] + 1e-4,
        joint_ranges[:, 1] - 1e-4
    )
    state["q_ref"] = q_ref

    q = data.qpos[qadr]
    qd = data.qvel[dofs]

    torque = (
        data.qfrc_bias[dofs]
        + KP * (q_ref - q)
        + KD * (qdot - qd)
    )

    data.ctrl[actuator_ids] = np.clip(
        torque,
        torque_limits[:, 0],
        torque_limits[:, 1]
    )
    data.ctrl[gripper_id] = state["gripper"]

    mujoco.mj_step(model, data)


# ------------------------------------------------------------
# 3. SMOOTH CARTESIAN MOTION
# ------------------------------------------------------------
def move_cartesian(viewer, target, duration, label):
    """Move along a smooth Cartesian line using a quintic time profile."""
    target = np.asarray(target, dtype=float)
    mujoco.mj_forward(model, data)
    start = data.site_xpos[site_id].copy()
    displacement = target - start

    steps = max(1, int(duration / dt))
    log_every = max(1, int(0.5 / dt))

    print(f"\n{label}")

    for i in range(steps):
        if not viewer.is_running():
            return False

        t = (i + 1) / steps

        # Quintic smoothstep: zero velocity and acceleration at endpoints.
        s = 10*t**3 - 15*t**4 + 6*t**5
        ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / duration

        desired_position = start + s * displacement
        desired_velocity = ds_dt * displacement

        dls_step(desired_position, desired_velocity)
        viewer.sync()

        if i % log_every == 0:
            print(
                f"EE={np.round(data.site_xpos[site_id], 3)} | "
                f"cube={np.round(data.xpos[cube_id], 4)}"
            )

        time.sleep(dt)

    return True


def hold_position(viewer, target, duration, label):
    """Hold the tool at a Cartesian position while the cube settles."""
    print(label)
    for _ in range(max(1, int(duration / dt))):
        if not viewer.is_running():
            return False
        dls_step(np.asarray(target, dtype=float), np.zeros(3))
        viewer.sync()
        time.sleep(dt)
    return True


def ramp_gripper(viewer, target_position, final_control, duration, label):
    """Open or close the gripper gradually while holding tool position."""
    start_control = state["gripper"]
    steps = max(1, int(duration / dt))
    print(label)

    for i in range(steps):
        if not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        state["gripper"] = (
            start_control + alpha * (final_control - start_control)
        )

        dls_step(np.asarray(target_position, dtype=float), np.zeros(3))
        viewer.sync()
        time.sleep(dt)

    state["gripper"] = final_control
    return True


# ------------------------------------------------------------
# 4. GRIPPER CONTACT CHECK
# ------------------------------------------------------------
def pad_contact_sides():
    sides = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        if cube_id not in (b1, b2):
            continue

        other_geom = g2 if b1 == cube_id else g1
        other_body = b2 if b1 == cube_id else b1

        geom_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
        ).lower()
        body_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, other_body
            ) or ""
        ).lower()

        combined = geom_name + " " + body_name

        if "left_pad" in combined:
            sides.add("left")
        if "right_pad" in combined:
            sides.add("right")

    return sides


def inside_tray(position):
    tray_pos = data.xpos[tray_id]
    dx = position[0] - tray_pos[0]
    dy = position[1] - tray_pos[1]
    return (
        abs(dx) <= 0.055
        and abs(dy) <= 0.055
        and 0.215 <= position[2] <= 0.29
    )


# ------------------------------------------------------------
# 5. INTEGRATED PICK-AND-PLACE
# ------------------------------------------------------------
lift_verified = False
place_verified = False
ok = True

with mujoco.viewer.launch_passive(model, data) as viewer:

    # A. Move above the cube.
    ok = move_cartesian(
        viewer, PREGRASP, 2.0, "1. Approach above cube"
    )
    if ok:
        ok = hold_position(
            viewer, PREGRASP, 0.5, "Settle above cube"
        )

    # B. Descend vertically.
    if ok:
        ok = move_cartesian(
            viewer, GRASP, 1.8, "2. Descend vertically to cube"
        )
    if ok:
        ok = hold_position(
            viewer, GRASP, 0.5, "Settle at grasp height"
        )

    # C. Close gripper.
    if ok:
        ok = ramp_gripper(
            viewer, GRASP, 255.0, 1.0, "3. Close gripper"
        )
    if ok:
        ok = hold_position(
            viewer, GRASP, 0.7, "Stabilize the grasp"
        )

    sides = pad_contact_sides()
    cube_after_close = data.xpos[cube_id].copy()
    close_shift = float(
        np.linalg.norm(cube_after_close[:2] - cube_start[:2])
    )

    print("\n=== GRASP CHECK ===")
    print("Pad contact:", sorted(sides))
    print("Cube horizontal shift during closing:", round(close_shift, 4), "m")

    if ok and (sides != {"left", "right"} or close_shift > 0.025):
        print("Grasp rejected. Stopping before lift.")
        ok = False

    # D. Lift STRAIGHT UP before any horizontal motion.
    if ok:
        ok = move_cartesian(
            viewer, LIFT, 2.0, "4. Lift vertically for tray clearance"
        )
    if ok:
        ok = hold_position(
            viewer, LIFT, 0.8, "Verify cube stays elevated"
        )

    mujoco.mj_forward(model, data)
    cube_after_lift = data.xpos[cube_id].copy()
    vertical_rise = float(cube_after_lift[2] - cube_start[2])
    lateral_shift = float(
        np.linalg.norm(cube_after_lift[:2] - cube_start[:2])
    )

    print("\n=== LIFT VERIFICATION ===")
    print("Cube position:", np.round(cube_after_lift, 4))
    print("Vertical rise:", round(vertical_rise, 4), "m")
    print("Lateral shift:", round(lateral_shift, 4), "m")

    lift_verified = bool(
        ok and vertical_rise >= 0.025 and lateral_shift <= 0.025
    )

    # E. Keep a high, constant tool height while carrying to the tray.
    if lift_verified:
        ok = move_cartesian(
            viewer, TRAY_HOVER, 2.5,
            "5. Carry horizontally above tray at clearance height"
        )

    # F. Lower vertically into the centre of the tray.
    if lift_verified and ok:
        ok = move_cartesian(
            viewer, TRAY_PLACE, 2.0,
            "6. Descend vertically into tray centre"
        )
    if lift_verified and ok:
        ok = hold_position(
            viewer, TRAY_PLACE, 0.5, "Settle over tray base"
        )

    # G. Release cube.
    if lift_verified and ok:
        ok = ramp_gripper(
            viewer, TRAY_PLACE, 0.0, 0.8, "7. Open gripper to release"
        )
    if lift_verified and ok:
        ok = hold_position(
            viewer, TRAY_PLACE, 0.8, "Let cube settle in tray"
        )

    # H. Retreat vertically instead of sweeping sideways along the wall.
    if lift_verified and ok:
        ok = move_cartesian(
            viewer, TRAY_HOVER, 1.8, "8. Retreat vertically above tray"
        )

    mujoco.mj_forward(model, data)
    cube_final = data.xpos[cube_id].copy()
    place_verified = bool(
        lift_verified and ok and inside_tray(cube_final)
    )

    print("\n=== FINAL SUMMARY ===")
    print("Lift verified:", lift_verified)
    print("Place verified:", place_verified)
    print("Initial cube position:", np.round(cube_start, 4))
    print("Final cube position:", np.round(cube_final, 4))

    if lift_verified and place_verified:
        print("RESULT: PICK-AND-PLACE SUCCESS")
    else:
        print("RESULT: NOT VERIFIED — inspect the failed stage.")

    print("Close the viewer when you finish inspecting the robot.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)

print("Simulation finished.")
