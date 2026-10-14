

import os
import csv
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "robot_descriptions",
    "franka_emika_panda",
    "panda.xml"
)

PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD FRANKA PANDA MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = []

for name in joint_names:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    joint_ids.append(jid)

qpos_addresses = np.array(
    [model.jnt_qposadr[jid] for jid in joint_ids]
)
dof_addresses = np.array(
    [model.jnt_dofadr[jid] for jid in joint_ids]
)
joint_ranges = np.array(
    [model.jnt_range[jid] for jid in joint_ids]
)

# Use the hand body as the initial TCP reference.
# We will refine this to the actual gripper TCP in the
# pick-and-place scene.
ee_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)
if ee_id < 0:
    raise ValueError("Could not find body named 'hand'.")

print("Model loaded:", MODEL_PATH)
print("Arm joints:", len(joint_ids))
print("End-effector reference: hand")

# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_pose(q):
    """Return hand position and rotation for a 7-joint vector."""
    data.qpos[qpos_addresses] = q
    mujoco.mj_forward(model, data)

    position = data.xpos[ee_id].copy()
    rotation = data.xmat[ee_id].reshape(3, 3).copy()
    return position, rotation


def rotation_error_vector(R_current, R_target):
    """Small-angle rotation error, expressed in world coordinates."""
    R_err = R_target @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])


# ============================================================
# TASK WORKSPACE: RANDOM JOINT SAMPLING
# ============================================================

N_SAMPLES = 20000
rng = np.random.default_rng(42)

positions = np.zeros((N_SAMPLES, 3))
orientations = np.zeros((N_SAMPLES, 3, 3))

for i in range(N_SAMPLES):
    q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
    positions[i], orientations[i] = forward_pose(q)

np.savez_compressed(
    os.path.join(RESULTS_DIR, "workspace_samples.npz"),
    positions=positions,
    orientations=orientations
)

# Summary statistics
mins = positions.min(axis=0)
maxs = positions.max(axis=0)
means = positions.mean(axis=0)

with open(
    os.path.join(RESULTS_DIR, "workspace_summary.csv"),
    "w",
    newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["Quantity", "X_m", "Y_m", "Z_m"])
    writer.writerow(["Minimum", *mins])
    writer.writerow(["Maximum", *maxs])
    writer.writerow(["Mean", *means])

# Plot task workspace
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2],
    s=1, alpha=0.25
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Franka Panda Task Workspace")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "task_workspace.png"),
    dpi=200
)
plt.close(fig)

# 2D top-view projection
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.25)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Task Workspace: XY Projection")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "task_workspace_xy.png"),
    dpi=200
)
plt.close(fig)

print(f"Task workspace samples: {N_SAMPLES}")
print("X range:", mins[0], "to", maxs[0])
print("Y range:", mins[1], "to", maxs[1])
print("Z range:", mins[2], "to", maxs[2])

# ============================================================
# NUMERICAL INVERSE KINEMATICS
# ============================================================

def solve_ik(target_position, target_yaw, q_initial,
             max_iterations=250, position_tolerance=0.015,
             orientation_tolerance=0.15):
    """
    Damped least-squares IK with position and yaw constraints.

    The hand's roll/pitch are not fully constrained. The
    controller targets position plus world-frame yaw.
    Returns a joint solution or None.
    """

    q = np.clip(
        q_initial.copy(),
        joint_ranges[:, 0] + 1e-5,
        joint_ranges[:, 1] - 1e-5
    )

    # Target yaw is imposed as a rotation about world Z.
    # Other orientation components remain unconstrained.
    for _ in range(max_iterations):
        data.qpos[qpos_addresses] = q
        mujoco.mj_forward(model, data)

        current_position = data.xpos[ee_id].copy()
        current_rotation = data.xmat[ee_id].reshape(3, 3).copy()

        position_error = target_position - current_position

        current_yaw = np.arctan2(
            current_rotation[1, 0], current_rotation[0, 0]
        )
        yaw_error = np.arctan2(
            np.sin(target_yaw - current_yaw),
            np.cos(target_yaw - current_yaw)
        )

        if (np.linalg.norm(position_error) < position_tolerance
                and abs(yaw_error) < orientation_tolerance):
            return q.copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

        Jp = jacp[:, dof_addresses]
        J_yaw = jacr[2:3, dof_addresses]

        J = np.vstack([Jp, 0.25 * J_yaw])
        error = np.concatenate([position_error, [0.25 * yaw_error]])

        damping = 0.04
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(4), error
        )

        # Limit each update to prevent unstable steps.
        step_scale = max(1.0, np.linalg.norm(dq) / 0.15)
        q += dq / step_scale

        # Keep solutions within joint limits.
        q = np.clip(
            q,
            joint_ranges[:, 0] + 1e-5,
            joint_ranges[:, 1] - 1e-5
        )

    return None


# ============================================================
# DEXTEROUS WORKSPACE APPROXIMATION
# ============================================================

# Use a manageable subset of the task-workspace samples as
# candidate positions. Test several yaw orientations at each.
N_CANDIDATES = 160
N_YAW = 8

candidate_indices = rng.choice(
    N_SAMPLES, size=N_CANDIDATES, replace=False
)
candidate_positions = positions[candidate_indices]

yaw_targets = np.linspace(-np.pi, np.pi, N_YAW, endpoint=False)

dexterous_positions = []
orientation_success_counts = []
dexterous_yaw_ranges = []

for i, target in enumerate(candidate_positions):
    success_yaws = []

    # Try multiple seeds to reduce false failures.
    for yaw_target in yaw_targets:
        solved = False

        for seed_index in rng.choice(N_SAMPLES, size=3, replace=False):
            seed_q = rng.uniform(
                joint_ranges[:, 0], joint_ranges[:, 1]
            )
            solution = solve_ik(target, yaw_target, seed_q)

            if solution is not None:
                solved = True
                break

        if solved:
            success_yaws.append(yaw_target)

    count = len(success_yaws)
    orientation_success_counts.append(count)

    # A candidate is labelled dexterous when at least half
    # of the tested yaw targets are feasible.
    if count >= N_YAW // 2:
        dexterous_positions.append(target)

    if success_yaws:
        dexterous_yaw_ranges.append(
            [target[0], target[1], target[2],
             min(success_yaws), max(success_yaws), count]
        )

    if (i + 1) % 10 == 0:
        print(f"IK workspace test: {i + 1}/{N_CANDIDATES}")

dexterous_positions = np.asarray(dexterous_positions).reshape(-1, 3)

np.savetxt(
    os.path.join(RESULTS_DIR, "dexterous_workspace.csv"),
    np.asarray(dexterous_yaw_ranges),
    delimiter=",",
    header="x_m,y_m,z_m,min_success_yaw_rad,max_success_yaw_rad,successful_yaw_count",
    comments=""
)

# Plot candidate points coloured by number of feasible yaws.
counts = np.asarray(orientation_success_counts)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    candidate_positions[:, 0],
    candidate_positions[:, 1],
    candidate_positions[:, 2],
    c=counts,
    s=18,
    alpha=0.85
)
fig.colorbar(scatter, ax=ax, label="Feasible yaw targets")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Franka Dexterous Workspace Approximation")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "dexterous_workspace.png"),
    dpi=200
)
plt.close(fig)

# Plot the subset meeting the chosen dexterity criterion.
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2],
    s=1, alpha=0.05, label="Task workspace samples"
)
if len(dexterous_positions) > 0:
    ax.scatter(
        dexterous_positions[:, 0],
        dexterous_positions[:, 1],
        dexterous_positions[:, 2],
        s=18, alpha=0.85, label="Dexterous subset"
    )
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Task and Dexterous Workspace")
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "task_vs_dexterous_workspace.png"),
    dpi=200
)
plt.close(fig)

# Save summary metrics
with open(
    os.path.join(RESULTS_DIR, "dexterous_summary.txt"),
    "w"
) as f:
    f.write(f"Task workspace samples: {N_SAMPLES}\n")
    f.write(f"Candidate positions tested: {N_CANDIDATES}\n")
    f.write(f"Yaw targets per position: {N_YAW}\n")
    f.write("Yaw targets: " + ", ".join(
        f"{v:.3f}" for v in yaw_targets
    ) + " rad\n")
    f.write(
        f"Dexterous candidates meeting criterion: "
        f"{len(dexterous_positions)}\n"
    )
    f.write(
        "Criterion: at least half of tested yaw targets "
        "solved within IK tolerances and joint limits.\n"
    )
    f.write(
        "Note: results are approximate and depend on sampling, "
        "IK convergence, seed choice, and the hand-body reference.\n"
    )

print("\nWorkspace analysis complete.")
print("Plots saved in:", PLOTS_DIR)
print("Numerical results saved in:", RESULTS_DIR)
print("Review dexterous_summary.txt before using the results.")

