from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mujoco.viewer
import mink


# ============================================================
# 1. PREPARE A PHYSICALLY MORE REASONABLE CUBE
# ============================================================

SOURCE_SCENE = Path(
    "scenes/heal_pick_place_gripper_front_table.xml"
)
SCENE = Path(
    "scenes/heal_pick_place_gripper_front_table_50g.xml"
)

tree = ET.parse(SOURCE_SCENE)
cube_body_xml = tree.find(".//worldbody/body[@name='cube']")

if cube_body_xml is None:
    raise RuntimeError("Could not find body named 'cube'.")

inertial = cube_body_xml.find("inertial")
if inertial is None:
    raise RuntimeError("Cube inertial element was not found.")

# Cube side = 0.05 m, mass = 0.05 kg.
# Solid-cube inertia: I = m*a^2/6.
cube_mass = 0.05
cube_inertia = cube_mass * (0.05 ** 2) / 6.0

inertial.set("mass", str(cube_mass))
inertial.set(
    "diaginertia",
    f"{cube_inertia:.9g} {cube_inertia:.9g} {cube_inertia:.9g}"
)

# Raise cube friction to improve contact grip.
for geom in cube_body_xml.findall("geom"):
    geom.set("friction", "2.0 0.01 0.001")

# Improve friction specifically on the Robotiq finger-pad surfaces.
for pad_name in ("left_pad", "right_pad"):
    pad_body = tree.find(f".//body[@name='{pad_name}']")
    if pad_body is not None:
        for geom in pad_body.findall("geom"):
            geom.set("friction", "2.0 0.01 0.001")

tree.write(SCENE, encoding="utf-8", xml_declaration=True)

print("Prepared scene:", SCENE)
print(f"Cube mass: {cube_mass:.3f} kg")
print(f"Cube diagonal inertia: {cube_inertia:.8g} kg m^2")


# ============================================================
# 2. LOAD MUJOCO MODEL AND DEFINE ROBOT
# ============================================================

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

dt = float(model.opt.timestep)

joint_names = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]
actuator_names = [
    "turret", "shoulder", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

joint_ids = np.array([model.joint(n).id for n in joint_names])
arm_actuator_ids = np.array(
    [model.actuator(n).id for n in actuator_names]
)

qadr = model.jnt_qposadr[joint_ids]
dofs = model.jnt_dofadr[joint_ids]
arm_limits = model.actuator_ctrlrange[arm_actuator_ids]

gripper_id = model.actuator("fingers_actuator").id
site_id = model.site("right_center").id
cube_id = model.body("cube").id
tray_id = model.body("tray").id

# Preserve the orientation already tested with the HEAL model.
TOOL_ROTATION = np.diag([-1.0, 1.0, -1.0])

# Positions refer to the right_center end-effector site.
PREGRASP = np.array([0.45, 0.10, 0.45])
GRASP = np.array([0.45, 0.10, 0.36])
LIFT = np.array([0.45, 0.10, 0.48])
TRAY_HOVER = np.array([0.35, -0.15, 0.48])
TRAY_PLACE = np.array([0.35, -0.15, 0.375])

# Control 0 = open, 255 = close.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0
gripper_command = GRIPPER_OPEN

# Joint-space tracking gains; actuator commands are torque-limited.
KP = 100.0
KD = 10.0

q_home = data.qpos[qadr].copy()
cube_start = data.xpos[cube_id].copy()

print("\n=== HEAL INTEGRATED PICK-AND-PLACE ===")
print("Scene:", SCENE)
print("Robot orientation: unchanged")
print("Initial cube position:", np.round(cube_start, 4))
print("Tray position:", np.round(data.xpos[tray_id], 4))


# ============================================================
# 3. MINK INVERSE KINEMATICS
# ============================================================

def solve_target(position):
    """Return the six arm joint positions for an EE target."""
    cfg = mink.Configuration(model)
    cfg.update()

    task = mink.FrameTask(
        frame_name="right_center",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
    )

    task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(TOOL_ROTATION),
            np.asarray(position, dtype=float),
        )
    )

    for _ in range(2500):
        cfg.update()
        velocity = mink.solve_ik(
            cfg,
            [task],
            dt,
            solver="daqp",
        )
        cfg.integrate_inplace(velocity, dt)

    # Verify the target in an independent MuJoCo data state.
    check = mujoco.MjData(model)
    check.qpos[:] = cfg.q
    mujoco.mj_forward(model, check)

    error = float(
        np.linalg.norm(position - check.site_xpos[site_id])
    )
    print(
        "IK target", np.round(position, 3),
        "| position error =", f"{error:.5f} m"
    )

    if not np.isfinite(error) or error > 0.01:
        raise RuntimeError(
            f"IK failed for target {position}; error={error:.4f} m"
        )

    return cfg.q[qadr].copy()


# Calculate targets before moving. If IK fails, no trial starts.
q_pre = solve_target(PREGRASP)
q_grasp = solve_target(GRASP)
q_lift = solve_target(LIFT)
q_tray_hover = solve_target(TRAY_HOVER)
q_tray_place = solve_target(TRAY_PLACE)


# ============================================================
# 4. ARM CONTROLLER AND TRAJECTORIES
# ============================================================

def control_step(q_reference):
    """Apply gravity-compensated joint-space PD control."""
    mujoco.mj_step1(model, data)

    q = data.qpos[qadr]
    qd = data.qvel[dofs]

    torque = (
        data.qfrc_bias[dofs]
        + KP * (q_reference - q)
        - KD * qd
    )

    data.ctrl[arm_actuator_ids] = np.clip(
        torque,
        arm_limits[:, 0],
        arm_limits[:, 1],
    )
    data.ctrl[gripper_id] = gripper_command

    mujoco.mj_step2(model, data)


def move_segment(viewer, q_from, q_to, duration, label):
    """Move smoothly between two joint configurations."""
    print("\n" + label)
    steps = max(1, int(duration / dt))
    log_every = max(1, int(1.0 / dt))

    for i in range(steps):
        if not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        blend = 3.0 * alpha**2 - 2.0 * alpha**3
        q_reference = q_from + blend * (q_to - q_from)

        control_step(q_reference)
        viewer.sync()

        if i % log_every == 0:
            print(
                f"t={i * dt:.1f}s | "
                f"EE={np.round(data.site_xpos[site_id], 3)} | "
                f"cube={np.round(data.xpos[cube_id], 4)}"
            )

        time.sleep(dt)

    return True


def hold(viewer, q_reference, duration, label):
    """Hold a joint target while the simulation runs."""
    print(label)

    for _ in range(max(1, int(duration / dt))):
        if not viewer.is_running():
            return False

        control_step(q_reference)
        viewer.sync()
        time.sleep(dt)

    return True


def close_gripper(viewer, q_reference, duration):
    """Ramp the gripper control from open to fully closed."""
    global gripper_command

    print("\nClosing gripper gradually to control 255")
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return False

        gripper_command = GRIPPER_CLOSED * (i + 1) / steps
        control_step(q_reference)
        viewer.sync()
        time.sleep(dt)

    gripper_command = GRIPPER_CLOSED
    return hold(viewer, q_reference, 0.8, "Stabilize closed gripper")


def open_gripper(viewer, q_reference, duration=0.6):
    """Ramp the gripper control from closed to open."""
    global gripper_command

    print("\nOpening gripper to release cube")
    start = gripper_command
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        gripper_command = start * (1.0 - alpha)
        control_step(q_reference)
        viewer.sync()
        time.sleep(dt)

    gripper_command = GRIPPER_OPEN
    return hold(viewer, q_reference, 0.6, "Allow cube to settle")


# ============================================================
# 5. CONTACT AND PLACEMENT CHECKS
# ============================================================

def pad_contact_sides():
    """Report whether both Robotiq pads contact the cube."""
    sides = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])

        if cube_id not in (body1, body2):
            continue

        other_geom = geom2 if body1 == cube_id else geom1
        geom_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
        ).lower()

        other_body = body2 if body1 == cube_id else body1
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


def cube_in_tray(position):
    """Check cube centre against the tray's approximate inner footprint."""
    tray_position = data.xpos[tray_id]
    dx = float(position[0] - tray_position[0])
    dy = float(position[1] - tray_position[1])

    inside_xy = abs(dx) <= 0.045 and abs(dy) <= 0.045
    inside_z = 0.205 <= position[2] <= 0.30

    return inside_xy and inside_z


# ============================================================
# 6. COMPLETE PICK-AND-PLACE TRIAL
# ============================================================

lift_verified = False
place_verified = False
cube_after_lift = cube_start.copy()
cube_final = cube_start.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Stage 1: approach above cube.
    ok = move_segment(
        viewer, data.qpos[qadr].copy(), q_pre,
        1.4, "1. APPROACH ABOVE CUBE"
    )
    if ok:
        ok = hold(viewer, q_pre, 0.5, "Settle above cube")

    # Stage 2: lower to the selected grasp pose.
    if ok:
        ok = move_segment(
            viewer, data.qpos[qadr].copy(), q_grasp,
            1.6, "2. LOWER TO GRASP"
        )
    if ok:
        ok = hold(viewer, q_grasp, 0.5, "Settle at grasp height")

    # Stage 3: close and require contact from both finger pads.
    if ok:
        ok = close_gripper(viewer, q_grasp, 0.8)

    if ok:
        mujoco.mj_forward(model, data)
        sides = pad_contact_sides()
        cube_after_close = data.xpos[cube_id].copy()
        cube_shift_close = float(
            np.linalg.norm(cube_after_close[:2] - cube_start[:2])
        )

        print("\n=== GRASP CHECK ===")
        print("Pad contact sides:", sorted(sides))
        print("Cube position:", np.round(cube_after_close, 4))
        print("Horizontal cube shift (m):", round(cube_shift_close, 4))

        if sides != {"left", "right"}:
            print(
                "Grasp not confirmed: both finger pads are not "
                "detected at the cube."
            )
            ok = False

        if cube_shift_close > 0.025:
            print("Grasp rejected: cube moved too far during closing.")
            ok = False

    # Stage 4: lift a small amount and verify the cube follows.
    if ok:
        ok = move_segment(
            viewer, data.qpos[qadr].copy(), q_lift,
            1.4, "4. LIFT CUBE"
        )

    if ok:
        ok = hold(viewer, q_lift, 0.7, "Verify the cube stays lifted")

    if ok:
        mujoco.mj_forward(model, data)
        cube_after_lift = data.xpos[cube_id].copy()

        vertical_rise = float(
            cube_after_lift[2] - cube_start[2]
        )
        lateral_shift = float(
            np.linalg.norm(cube_after_lift[:2] - cube_start[:2])
        )

        print("\n=== LIFT VERIFICATION ===")
        print("Cube position:", np.round(cube_after_lift, 4))
        print("Vertical rise (m):", round(vertical_rise, 4))
        print("Lateral shift (m):", round(lateral_shift, 4))

        lift_verified = (
            vertical_rise >= 0.02
            and lateral_shift <= 0.025
        )

    if not lift_verified:
        print("\nPICK FAILED OR NOT VERIFIED.")
        print("Tray transfer stopped; inspect the grasp before retrying.")

        # Avoid continuing to drag the cube across the scene.
        if viewer.is_running():
            gripper_command = GRIPPER_OPEN
            hold(viewer, data.qpos[qadr].copy(), 0.5,
                 "Release after failed lift test")

    else:
        # Stage 5: carry cube above the tray.
        ok = move_segment(
            viewer, data.qpos[qadr].copy(), q_tray_hover,
            1.6, "5. CARRY ABOVE TRAY"
        )

        # Confirm the cube is still elevated before lowering.
        mujoco.mj_forward(model, data)
        cube_before_place = data.xpos[cube_id].copy()

        if ok and cube_before_place[2] < cube_start[2] + 0.015:
            print("Cube slipped during transfer; stopping before release.")
            ok = False

        # Stage 6: lower cube into tray.
        if ok:
            ok = move_segment(
                viewer, data.qpos[qadr].copy(), q_tray_place,
                1.5, "6. LOWER INTO TRAY"
            )

        if ok:
            ok = hold(
                viewer, q_tray_place, 0.4,
                "Settle above tray base"
            )

        # Stage 7: release.
        if ok:
            ok = open_gripper(viewer, q_tray_place)

        # Let the released cube settle before evaluating placement.
        if ok:
            for _ in range(int(0.7 / dt)):
                if not viewer.is_running():
                    ok = False
                    break
                control_step(q_tray_place)
                viewer.sync()
                time.sleep(dt)

        # Stage 8: retreat only after release.
        if ok:
            ok = move_segment(
                viewer, data.qpos[qadr].copy(), q_tray_hover,
                1.4, "8. RETREAT ABOVE TRAY"
            )

        mujoco.mj_forward(model, data)
        cube_final = data.xpos[cube_id].copy()
        place_verified = bool(ok and cube_in_tray(cube_final))

        print("\n=== PLACE RESULTS ===")
        print("Final cube position:", np.round(cube_final, 4))
        print("Cube within tray region:", place_verified)

    print("\n=== FINAL SUMMARY ===")
    print("Lift verified:", lift_verified)
    print("Place verified:", place_verified)
    print("Initial cube position:", np.round(cube_start, 4))
    print("Final cube position:", np.round(data.xpos[cube_id], 4))
    if lift_verified and place_verified:
        print("RESULT: PICK-AND-PLACE SUCCESS")
    else:
        print("RESULT: NOT YET SUCCESSFUL; use output to diagnose.")
    print("Inspect the viewer, then close it to finish.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)

print("Simulation finished.")
