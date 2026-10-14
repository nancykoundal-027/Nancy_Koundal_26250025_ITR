
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

BASE = os.path.expanduser("~/ITR_LAB3_FRANKA")
MODEL_PATH = os.path.join(
    BASE,
    "robot_descriptions",
    "franka_emika_panda",
    "pick_place_scene.xml",
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in joint_names
]

if any(jid < 0 for jid in joint_ids):
    raise ValueError("Could not find all seven Panda arm joints.")

qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids])

actuator_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"
    )
    for i in range(1, 8)
]

if any(aid < 0 for aid in actuator_ids):
    raise ValueError("Could not find all seven arm actuators.")

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)
cube_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "cube"
)

if hand_id < 0 or cube_id < 0:
    raise ValueError("Could not find the hand or cube body.")

# Start from the model's home keyframe if available.
key_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)

if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    mujoco.mj_resetData(model, data)

# Restore the cube to its intended initial position.
cube_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
)

if cube_joint_id >= 0:
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = np.array(
        [0.45, 0.0, 0.35]
    )
    data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

mujoco.mj_forward(model, data)

# Jacobian-based position IK for the hand body.
def solve_ik(target, iterations=250):
    for _ in range(iterations):
        mujoco.mj_forward(model, data)

        error = target - data.xpos[hand_id].copy()
        if np.linalg.norm(error) < 0.015:
            return True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

        J = jacp[:, dof_ids]
        damping = 0.05
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3), error
        )


        q = data.qpos[qpos_ids].copy()
        dq = np.clip(dq, -0.005, 0.005)
        q += dq
        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)

        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
    return np.linalg.norm(target - data.xpos[hand_id]) < 0.03


# Initial target: above the cube.
cube_pos = data.xpos[cube_id].copy()
target = cube_pos + np.array([0.0, 0.0, 0.15])

print("Robot loaded:", model.nbody, "bodies")
print("Cube position:", cube_pos)
print("Target above cube:", target)
print("Running position IK test...")

initial_qpos = data.qpos.copy()
success = solve_ik(target)
print("IK target reached:", success)
print("Hand position:", data.xpos[hand_id].copy())

# Hold the resulting arm configuration using joint position actuators.
# Move gradually from the home position to the IK solution.
if success:
    desired = data.qpos[qpos_ids].copy()

    # Save the initial joint configuration.
    start_q = initial_qpos[qpos_ids].copy()

    # Restore the robot to its initial configuration.
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)


    duration = 15.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.monotonic()

        while viewer.is_running():
            elapsed = time.monotonic() - start_time
            alpha = min(elapsed / duration, 1.0)

            # Smooth movement: start slowly, accelerate gently,
            # and slow down near the final position.
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3
            command = start_q + smooth_alpha * (desired - start_q)

            for aid, q in zip(actuator_ids, command):
                data.ctrl[aid] = q

            # Keep the gripper actuator at its current command.

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

