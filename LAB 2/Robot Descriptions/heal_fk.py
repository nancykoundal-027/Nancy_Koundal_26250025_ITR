
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load HEAL robot
model = mujoco.MjModel.from_xml_path(
    "single_arm_heal_effort_actuation_rs_mj.xml"
)
data = mujoco.MjData(model)

joint_names = [f"joint_{i}" for i in range(1, 7)]
actuator_names = [
    "turret", "shoulder", "elbow",
    "wrist_1", "wrist_2", "wrist_3"
]
body_names = [
    "link_1", "link_2", "link_3",
    "link_4", "link_5", "end_effector"
]

joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    for n in joint_names
]
actuator_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    for n in actuator_names
]
body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    for n in body_names
]

if any(i < 0 for i in joint_ids + actuator_ids + body_ids):
    raise ValueError("A joint, actuator, or body name was not found. Check the XML names.")

qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]

# Make actuator control sliders represent joint angles in radians.
for aid in actuator_ids:
    model.actuator_ctrllimited[aid] = 1
    model.actuator_ctrlrange[aid] = [-np.pi, np.pi]
    data.ctrl[aid] = 0.0

np.set_printoptions(precision=4, suppress=True)

def print_fk():
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )
    base_R = data.xmat[base_id].reshape(3, 3)
    base_p = data.xpos[base_id]

    print("\nJoint angles (degrees):",
          np.round(np.rad2deg([data.qpos[q] for q in qpos_ids]), 2))

    for i, body_id in enumerate(body_ids, start=1):
        Rw = data.xmat[body_id].reshape(3, 3)
        pw = data.xpos[body_id]

        T = np.eye(4)
        T[:3, :3] = base_R.T @ Rw
        T[:3, 3] = base_R.T @ (pw - base_p)

        print(f"\nHomogeneous Transformation T_0_{i}:")
        print(T)
        print("Position (m):", T[:3, 3])

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_center"
    )
    if site_id >= 0:
        Rw = data.site_xmat[site_id].reshape(3, 3)
        pw = data.site_xpos[site_id]
        T = np.eye(4)
        T[:3, :3] = base_R.T @ Rw
        T[:3, 3] = base_R.T @ (pw - base_p)
        print("\nEnd-effector T_0_EE:")
        print(T)
        print("End-effector position (m):", T[:3, 3])

last_q = np.full(6, np.inf)
last_print = 0

# Direct kinematic control: no torque-driven physics stepping
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Read slider controls from the viewer
        viewer.sync()

        with viewer.lock():
            # Treat each actuator control as a desired joint angle
            for i, aid in enumerate(actuator_ids):
                data.qpos[qpos_ids[i]] = data.ctrl[aid]

            mujoco.mj_forward(model, data)
            q = np.array([data.qpos[q] for q in qpos_ids])

        # Print only when angles have changed
        if np.max(np.abs(q - last_q)) > np.deg2rad(0.5):
            if time.time() - last_print > 0.25:
                print_fk()
                last_q = q.copy()
                last_print = time.time()

        viewer.sync()
        time.sleep(0.02)

