
import time
import numpy as np
import mujoco
import mujoco.viewer

# Load your existing Franka model
model = mujoco.MjModel.from_xml_path("panda.xml")
data = mujoco.MjData(model)

# Identify the seven robot joints and their actuators
joint_names = [f"joint{i}" for i in range(1, 8)]
actuator_names = [f"actuator{i}" for i in range(1, 8)]

joint_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    ) for name in joint_names
]
actuator_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    ) for name in actuator_names
]

if any(j < 0 for j in joint_ids) or any(a < 0 for a in actuator_ids):
    raise ValueError("Could not find all seven joints and actuators")

qpos_ids = [int(model.jnt_qposadr[j]) for j in joint_ids]
body_ids = [int(model.jnt_bodyid[j]) for j in joint_ids]

# Initial joint angles, in degrees
initial_angles = [0, -30, 0, -120, 0, 90, 45]

# Set initial robot configuration and actuator targets
for qadr, aid, angle in zip(qpos_ids, actuator_ids, initial_angles):
    rad = np.deg2rad(angle)
    data.qpos[qadr] = rad
    data.ctrl[aid] = rad

mujoco.mj_forward(model, data)

def get_transform(body_id):
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T

def print_fk():
    print("\033[2J\033[H", end="")  # Clear terminal display
    print("FRANKA 7-DOF LIVE FORWARD KINEMATICS")
    print("-" * 55)

    for i, qadr in enumerate(qpos_ids):
        angle = np.rad2deg(data.qpos[qadr])
        print(f"Joint {i+1} angle: {angle:8.3f} degrees")

    for i, body_id in enumerate(body_ids):
        T = get_transform(body_id)
        print(f"\nHomogeneous Transformation T_0_{i+1}:")
        print(np.array2string(T, precision=4, suppress_small=True))
        print("Position (m):", np.round(T[:3, 3], 4))

    T_ee = get_transform(body_ids[-1])
    print("\nLast joint body position (m):")
    print(np.round(T_ee[:3, 3], 4))
    print("\nChange actuator sliders in MuJoCo to update these values.")

# Open viewer and continuously simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print = 0.0

    while viewer.is_running():
        with viewer.lock():
            # Receive slider changes from the viewer
            viewer.sync()

            # Advance physics using actuator controls
            mujoco.mj_step(model, data)

            # Refresh the viewer
            viewer.sync()

            # Refresh terminal output every 0.5 seconds
            now = time.monotonic()
            if now - last_print >= 0.5:
                print_fk()
                last_print = now

        time.sleep(model.opt.timestep)

