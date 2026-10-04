import os
import time
import numpy as np

import mujoco
import mujoco.viewer


# --------------------------------------------------
# 1. Load TurtleBot3 Waffle Pi
# --------------------------------------------------

XML_PATH = os.path.expanduser(
    "~/robotis_mujoco_menagerie/robotis_tb3/"
    "scene_turtlebot3_waffle_pi.xml"
)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# --------------------------------------------------
# 2. Find wheel actuators
# --------------------------------------------------

left_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)

if left_id < 0 or right_id < 0:
    raise RuntimeError("Wheel actuators not found.")


print("Left actuator :", left_id)
print("Right actuator:", right_id)


# --------------------------------------------------
# 3. Find robot body
# --------------------------------------------------

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)

if body_id < 0:
    raise RuntimeError("Robot body 'base' not found.")

print("Robot body found: base")


# --------------------------------------------------
# 4. Robot speed
# --------------------------------------------------

speed = 5.0

left_cmd = 0.0
right_cmd = 0.0


# --------------------------------------------------
# 5. Keyboard control
# --------------------------------------------------

def key_callback(keycode):

    global left_cmd, right_cmd

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    print("Key pressed:", key)

    # Forward
    if key == "w":
        left_cmd = speed
        right_cmd = speed

    # Backward
    elif key == "s":
        left_cmd = -speed
        right_cmd = -speed

    # Turn left
    elif key == "a":
        left_cmd = -speed
        right_cmd = speed

    # Turn right
    elif key == "d":
        left_cmd = speed
        right_cmd = -speed

    # Stop
    elif key == " ":
        left_cmd = 0.0
        right_cmd = 0.0


# --------------------------------------------------
# 6. Create body-frame arrow
# --------------------------------------------------

def create_arrow(geom, position, direction, length, rgba):

    direction = np.asarray(direction, dtype=float)

    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return

    direction = direction / norm

    # Reference vector
    ref = np.array([0.0, 0.0, 1.0])

    # Avoid parallel vectors
    if abs(np.dot(direction, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    # Construct orthonormal basis
    x_axis = np.cross(ref, direction)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(direction, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = direction

    rotation = np.column_stack(
        (x_axis, y_axis, z_axis)
    )

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.035, 0.035, length]),
        np.asarray(position, dtype=float),
        rotation.flatten(),
        np.asarray(rgba, dtype=float)
    )


# --------------------------------------------------
# 7. Start MuJoCo viewer
# --------------------------------------------------

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("\n==============================")
    print("TURTLEBOT3 WAFFLE PI")
    print("==============================")
    print("W = Forward")
    print("S = Backward")
    print("A = Turn Left")
    print("D = Turn Right")
    print("SPACE = Stop")
    print("==============================\n")

    # Reserve 3 user geoms:
    # 0 = X axis
    # 1 = Y axis
    # 2 = Z axis

    viewer.user_scn.ngeom = 3

    last_matrix_print = 0.0

    while viewer.is_running():

        # --------------------------------------------------
        # Apply wheel commands
        # --------------------------------------------------

        data.ctrl[left_id] = left_cmd
        data.ctrl[right_id] = right_cmd

        mujoco.mj_step(model, data)


        # --------------------------------------------------
        # Robot position
        # --------------------------------------------------

        position = data.xpos[body_id].copy()


        # --------------------------------------------------
        # Rotation matrix
        # --------------------------------------------------

        R = data.xmat[body_id].reshape(3, 3).copy()


        # --------------------------------------------------
        # Body-frame axes expressed in world coordinates
        # --------------------------------------------------

        x_body = R[:, 0]
        y_body = R[:, 1]
        z_body = R[:, 2]


        # --------------------------------------------------
        # Put the body frame slightly above the robot
        # --------------------------------------------------

        frame_position = position + np.array(
            [0.0, 0.0, 0.20]
        )

        axis_length = 0.45


        # --------------------------------------------------
        # X axis
        # --------------------------------------------------

        create_arrow(
            viewer.user_scn.geoms[0],
            frame_position,
            x_body,
            axis_length,
            np.array([1.0, 0.0, 0.0, 1.0])
        )


        # --------------------------------------------------
        # Y axis
        # --------------------------------------------------

        create_arrow(
            viewer.user_scn.geoms[1],
            frame_position,
            y_body,
            axis_length,
            np.array([0.0, 1.0, 0.0, 1.0])
        )


        # --------------------------------------------------
        # Z axis
        # --------------------------------------------------

        create_arrow(
            viewer.user_scn.geoms[2],
            frame_position,
            z_body,
            axis_length,
            np.array([0.0, 0.0, 1.0, 1.0])
        )


        # --------------------------------------------------
        # Display rotation matrix in Terminal
        # --------------------------------------------------

        current_time = time.time()

        if current_time - last_matrix_print > 0.5:

            print("\033[H\033[J", end="")

            print("======================================")
            print("      TURTLEBOT3 WAFFLE PI")
            print("======================================")

            print("\nKeyboard:")
            print("W = Forward")
            print("S = Backward")
            print("A = Turn Left")
            print("D = Turn Right")
            print("SPACE = Stop")

            print("\nRobot Position:")
            print(
                f"x = {position[0]: .4f}, "
                f"y = {position[1]: .4f}, "
                f"z = {position[2]: .4f}"
            )

            print("\nBody Rotation Matrix R:")

            print(
                np.array2string(
                    R,
                    formatter={
                        "float_kind": lambda x: f"{x:8.4f}"
                    }
                )
            )

            print("\nBody X axis in World:")
            print(x_body)

            print("\nBody Y axis in World:")
            print(y_body)

            print("\nBody Z axis in World:")
            print(z_body)

            print("\n======================================")

            last_matrix_print = current_time


        # --------------------------------------------------
        # Update viewer
        # --------------------------------------------------

        viewer.sync()

        time.sleep(model.opt.timestep)
