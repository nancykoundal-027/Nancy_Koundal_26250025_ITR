import argparse
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

RESULT_FILES = {
    "mink": "results/mink_results_speedcap_test_25_episodes.json",
    "dls": "results/dls_results_speedcap_test_25_episodes.json",
    "qp": "results/qp_results_speedcap_test_25_episodes.json",
}

parser = argparse.ArgumentParser(
    description="Inspect the initial cube pose of a benchmark episode."
)
parser.add_argument(
    "--method", choices=["mink", "dls", "qp"], required=True
)
parser.add_argument("--episode", type=int)
parser.add_argument(
    "--list", action="store_true",
    help="List failed episodes for the selected method."
)
args = parser.parse_args()

result_path = Path(RESULT_FILES[args.method])
if not result_path.exists():
    raise SystemExit(f"Benchmark results not found: {result_path}")

rows = json.loads(result_path.read_text())
failed = [r for r in rows if not r.get("success", False)]

if args.list:
    print(f"Failed episodes for {args.method.upper()}:")
    for r in failed:
        print(
            f"Episode {r['episode']}: "
            f"{r.get('reason', 'Reason not recorded')}"
        )
    raise SystemExit(0)

if args.episode is None:
    raise SystemExit(
        "Specify --episode N, or use --list to list failed episodes."
    )

matches = [r for r in rows if int(r["episode"]) == args.episode]
if not matches:
    raise SystemExit(f"Episode {args.episode} was not found.")

episode = matches[0]
if episode.get("success", False):
    raise SystemExit(
        f"Episode {args.episode} succeeded in this benchmark. "
        "Choose an episode from the failed-episode list."
    )

position = np.asarray(episode["initial_cube_pose_xyz_m"], dtype=float)
yaw = float(episode["initial_cube_yaw_rad"])

scene = "robot_descriptions/franka_emika_panda/pick_place_scene.xml"
model = mujoco.MjModel.from_xml_path(scene)
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
else:
    mujoco.mj_resetData(model, data)

cube_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
)
if cube_joint_id < 0:
    raise SystemExit("Could not find the cube_free joint.")

qadr = int(model.jnt_qposadr[cube_joint_id])
quat_wxyz = R.from_euler("z", yaw).as_quat()
quat_wxyz = np.array([
    quat_wxyz[3], quat_wxyz[0],
    quat_wxyz[1], quat_wxyz[2]
])

data.qpos[qadr:qadr + 3] = position
data.qpos[qadr + 3:qadr + 7] = quat_wxyz
data.qvel[:] = 0
mujoco.mj_forward(model, data)

print(f"Method: {args.method.upper()}")
print(f"Episode: {args.episode}")
print(f"Failure reason: {episode.get('reason', 'Not recorded')}")
print(f"Initial cube position (m): {position}")
print(f"Initial cube yaw (rad): {yaw}")
print()
print(
    "Viewer shows the initial cube pose for this failed episode, "
    "not the robot's later state at the moment of failure."
)
print("Close the viewer window to return to Terminal.")

mujoco.viewer.launch(model, data)
