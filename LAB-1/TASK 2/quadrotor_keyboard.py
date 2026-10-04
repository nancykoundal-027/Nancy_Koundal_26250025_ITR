import time
import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# LOAD QUADROTOR
# ============================================================

XML_PATH = "quadrotor.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "quadrotor"
)

if body_id < 0:
    raise RuntimeError("Quadrotor body not found.")


# ============================================================
# KEYBOARD COMMANDS
# ============================================================

move_speed = 0.5

roll_rate = 0.8
pitch_rate = 0.8
yaw_rate = 0.8

vx = 0.0
vy = 0.0
vz = 0.0

roll_cmd = 0.0
pitch_cmd = 0.0
yaw_cmd = 0.0


def key_callback(keycode):

    global vx, vy, vz
    global roll_cmd, pitch_cmd, yaw_cmd

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    print("Key pressed:", key)

    # Translation
    if key == "w":
        vx = move_speed

    elif key == "s":
        vx = -move_speed

    elif key == "a":
        vy = move_speed

    elif key == "d":
        vy = -move_speed

    elif key == "r":
        vz = move_speed

    elif key == "f":
        vz = -move_speed

    # Yaw
    elif key == "q":
        yaw_cmd = yaw_rate

    elif key == "e":
        yaw_cmd = -yaw_rate

    # Roll
    elif key == "j":
        roll_cmd = roll_rate

    elif key == "l":
        roll_cmd = -roll_rate

    # Pitch
    elif key == "i":
        pitch_cmd = pitch_rate

    elif key == "k":
        pitch_cmd = -pitch_rate

    # Stop
    elif key == " ":
        vx = 0.0
        vy = 0.0
        vz = 0.0

        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_cmd = 0.0


# ============================================================
# ARROW FUNCTION
# ============================================================

def create_arrow(geom, position, direction, length, rgba):

    direction = np.asarray(direction, dtype=float)

    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return

    direction = direction / norm

    reference = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    axis1 = np.cross(reference, direction)
    axis1 /= np.linalg.norm(axis1)

    axis2 = np.cross(direction, axis1)
    axis2 /= np.linalg.norm(axis2)

    axis3 = direction

    rotation = np.column_stack(
        (axis1, axis2, axis3)
    )

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.03, 0.03, length]),
        np.asarray(position, dtype=float),
        rotation.flatten(),
        np.asarray(rgba, dtype=float)
    )


# ============================================================
# START VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("\n==========================================")
    print("             QUADROTOR")
    print("==========================================")
    print("W / S = Forward / Backward")
    print("A / D = Left / Right")
    print("R / F = Up / Down")
    print("Q / E = Yaw")
    print("J / L = Roll")
    print("I / K = Pitch")
    print("SPACE = Stop")
    print("==========================================")

    # 0,1,2 = world frame
    # 3,4,5 = body frame

    viewer.user_scn.ngeom = 6

    last_print = 0.0

    while viewer.is_running():

        dt = model.opt.timestep

        # ----------------------------------------------------
        # Current rotation matrix
        # ----------------------------------------------------

        quat = data.qpos[3:7].copy()

        rotation_flat = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation_flat,
            quat
        )

        R = rotation_flat.reshape(3, 3)

        x_body = R[:, 0]
        y_body = R[:, 1]
        z_body = R[:, 2]

        # ----------------------------------------------------
        # Body velocity -> world velocity
        # ----------------------------------------------------

        body_velocity = np.array([
            vx,
            vy,
            vz
        ])

        world_velocity = R @ body_velocity

        # ----------------------------------------------------
        # Translate
        # ----------------------------------------------------

        data.qpos[0] += world_velocity[0] * dt
        data.qpos[1] += world_velocity[1] * dt
        data.qpos[2] += world_velocity[2] * dt

        # ----------------------------------------------------
        # Roll + pitch + yaw
        # ----------------------------------------------------

        angular_velocity = np.array([
            roll_cmd,
            pitch_cmd,
            yaw_cmd
        ])

        angle = np.linalg.norm(
            angular_velocity
        ) * dt

        if angle > 1e-12:

            axis = (
                angular_velocity /
                np.linalg.norm(angular_velocity)
            )

            half_angle = angle / 2.0

            dq = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle)
            ])

            current_quat = data.qpos[3:7].copy()

            new_quat = np.zeros(4)

            mujoco.mju_mulQuat(
                new_quat,
                current_quat,
                dq
            )

            new_quat /= np.linalg.norm(new_quat)

            data.qpos[3:7] = new_quat

        # ----------------------------------------------------
        # Update MuJoCo
        # ----------------------------------------------------

        mujoco.mj_forward(model, data)

        position = data.xpos[body_id].copy()

        rotation_flat = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation_flat,
            data.qpos[3:7]
        )

        R = rotation_flat.reshape(3, 3)

        x_body = R[:, 0]
        y_body = R[:, 1]
        z_body = R[:, 2]

        # ----------------------------------------------------
        # World frame
        # ----------------------------------------------------

        world_origin = np.array([
            0.0,
            0.0,
            0.03
        ])

        world_x = np.array([1.0, 0.0, 0.0])
        world_y = np.array([0.0, 1.0, 0.0])
        world_z = np.array([0.0, 0.0, 1.0])

        create_arrow(
            viewer.user_scn.geoms[0],
            world_origin,
            world_x,
            0.50,
            np.array([1.0, 0.0, 0.0, 1.0])
        )

        create_arrow(
            viewer.user_scn.geoms[1],
            world_origin,
            world_y,
            0.50,
            np.array([0.0, 1.0, 0.0, 1.0])
        )

        create_arrow(
            viewer.user_scn.geoms[2],
            world_origin,
            world_z,
            0.50,
            np.array([0.0, 0.0, 1.0, 1.0])
        )

        # ----------------------------------------------------
        # Body frame
        # ----------------------------------------------------

        create_arrow(
            viewer.user_scn.geoms[3],
            position,
            x_body,
            0.40,
            np.array([1.0, 0.5, 0.0, 1.0])
        )

        create_arrow(
            viewer.user_scn.geoms[4],
            position,
            y_body,
            0.40,
            np.array([0.0, 1.0, 0.5, 1.0])
        )

        create_arrow(
            viewer.user_scn.geoms[5],
            position,
            z_body,
            0.40,
            np.array([0.7, 0.0, 1.0, 1.0])
        )

        # ----------------------------------------------------
        # Rotation matrix
        # ----------------------------------------------------

        now = time.time()

        if now - last_print > 0.1:

            print("\033[H\033[J", end="")

            print("==========================================")
            print("             QUADROTOR")
            print("==========================================")

            print("\nWORLD / INERTIAL FRAME W")
            print("Fixed at origin")

            print("\nBODY FRAME B")
            print("Attached to quadrotor")

            print("\nPosition:")
            print(f"x = {position[0]: .4f}")
            print(f"y = {position[1]: .4f}")
            print(f"z = {position[2]: .4f}")

            print("\nRotation Matrix  ^W R_B:")

            print(
                np.array2string(
                    R,
                    formatter={
                        "float_kind":
                        lambda x: f"{x:8.4f}"
                    }
                )
            )

            print("\nBody X axis in World:")
            print(x_body)

            print("\nBody Y axis in World:")
            print(y_body)

            print("\nBody Z axis in World:")
            print(z_body)

            print("\n==========================================")

            last_print = now

        viewer.sync()

        time.sleep(dt)
