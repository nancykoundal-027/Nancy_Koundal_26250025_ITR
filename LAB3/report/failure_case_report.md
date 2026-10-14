# Failure Case Analysis — Franka Panda Pick-and-Place

## Purpose

This report documents representative failed episodes from the randomized
25-episode benchmarks for Mink IK, Damped Least-Squares (DLS) IK, and
custom QP IK.

The screenshots show the initial scene configuration associated with each
selected failed episode. They are not screenshots of the exact instant
of failure.

## Case 1 — Mink IK: Episode 9

**Recorded failure:** IK did not converge during the pre-grasp phase.

The benchmark log records a position error of approximately 0.1524 m
and an orientation error of approximately 0.1927 rad at failure.

**Interpretation:** The controller did not bring the end effector within
the specified pre-grasp tolerances for this initial cube pose before the
iteration limit was reached.

**Evidence:** `../failure_screenshots/mink_ep09_initial_pose.png`

## Case 2 — DLS IK: Episode 14

**Recorded failure:** Grasp check failed because the cube did not rise by
at least 0.015 m.

**Interpretation:** The arm reached the commanded grasp and lift waypoints,
but the logged cube-height check did not confirm the required lift.
Waypoint convergence alone does not guarantee that the gripper has
securely grasped the object.

**Evidence:** `../failure_screenshots/dls_ep14_initial_pose.png`

## Case 3 — Custom QP IK: Episode 9

**Recorded failure:** IK did not converge during transit above the tray.
The recorded final errors were approximately 0.1034 m in position and
0.0243 rad in orientation.

**Interpretation:** The controller failed to reach the target above the
tray within the configured convergence limit. This prevented the episode
from completing the remaining placement sequence.

**Evidence:** `../failure_screenshots/qp_ep09_initial_pose.png`

## Common Observations

The failure examples illustrate three different limitations:

1. IK non-convergence during pre-grasp or transport.
2. Failure to verify a physical lift despite reaching the grasp and lift
   waypoints.
3. Incomplete task execution when an intermediate target is not reached.

These failures should be interpreted using the corresponding JSON logs,
including the failure reason, phase errors, initial cube pose, and
constraint metrics. The screenshots show the associated initial
configurations and should not be interpreted as the robot state at the
instant of failure.

## Limitations and Mitigation

Potential areas for further investigation include improving target
initialization, adjusting trajectory ramps and convergence tolerances,
checking gripper-object contact, and enforcing joint-position and
joint-velocity limits throughout execution.

Any proposed mitigation should be tested on the same randomized initial
poses and evaluated using task success, execution time, IK iterations,
and constraint-violation measurements.
