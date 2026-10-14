
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

BASE = os.path.expanduser("~/ITR_LAB3_FRANKA")
MODEL_PATH = os.path.join(
    BASE,
    "robot_descriptions",
    "franka_emika_panda",
    "pick_place_scene.xml",
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Start from the saved home configuration.
home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

# Find the arm joints, actuators, hand, and cube.
joint_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
    )
    for i in range(1, 8)
]
actuator_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"
    )
    for i in range(1, 8)
]
hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)
cube_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "cube"
)

if min(joint_ids + actuator_ids + [hand_id, cube_id]) < 0:
    raise ValueError("A required joint, actuator, hand, or cube was not found.")

qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids])
dof_ids = np.array([model.jnt_dofadr[j] for j in joint_ids])

# Mink configuration and hand position task.
configuration = mink.Configuration(model)
hand_task = mink.FrameTask(
    frame_name="hand",
    frame_type="body",
    position_cost=1.0,
    orientation_cost=0.0,
    lm_damping=1.0,
)

configuration.update(data.qpos)

# Start at the current hand position to prevent an initial jump.
start = data.xpos[hand_id].copy()
target = np.array([0.45, 0.0, 0.50])

hand_task.set_target(mink.SE3.from_translation(target))

# Preserve the initial arm and gripper commands.
hold_q = data.qpos[qpos_ids].copy()
for aid, q in zip(actuator_ids, hold_q):
    data.ctrl[aid] = q

print("MuJoCo loaded successfully.")
print("Initial hand position:", start)
print("Cube position:", data.xpos[cube_id].copy())
print("Target above cube:", target)
print("The cube will not be grasped in this test.")

# Smoothly move the target above the cube.
duration = 3.0
elapsed = 0.0
dt = model.opt.timestep
q_des = data.qpos[qpos_ids].copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        configuration.update(data.qpos)

        velocity = mink.solve_ik(
            configuration,
            [hand_task],
            model.opt.timestep,
            solver="daqp"
        )

        # Limit joint velocities for smooth motion
        velocity = np.clip(velocity, -0.5, 0.5)

        # Accumulate desired joint positions
        q_des += velocity[dof_ids] * dt

        # Apply desired joint positions to the arm 
        for aid, q in zip(actuator_ids, q_des):
            data.ctrl[aid] = q

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
