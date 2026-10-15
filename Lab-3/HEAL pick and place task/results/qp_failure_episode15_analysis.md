# QP-IK Failure Analysis — Episode 15

## Test details
- Method: QP-IK
- Episode: 15
- Seed: 12014
- Initial cube position: [0.5037, 0.1725, 0.225] m
- Final cube position: [0.5034, 0.1303, 0.2244] m

## Observed failure
The robot approached and descended toward the cube, but the grasp
verification detected no finger-pad contact. During gripper closure,
the cube moved horizontally by 0.0422 m (4.22 cm). The controller
rejected the grasp and stopped before lifting.

## Verification
- Lift verified: False
- Place verified: False
- Vertical rise: -0.0006 m
- Lateral shift during closure: 0.0422 m
- Result: PICK-AND-PLACE NOT VERIFIED

## Likely cause
The gripper likely missed the cube sides at this starting position
and pushed the cube during closure. The contact and displacement
measurements support this explanation.

## Proposed improvement
Improve grasp-target alignment and verify contact on both finger pads
before initiating the lift. Repeat the same starting configuration
after adjusting the grasp pose to determine whether the failure is
corrected.

## Evidence files
- Screenshot: results/qp_failure_episode15.png
- Log: results/qp_failure_replay_ep15.log
- Original stress-test log:
  results/heal_wide_stress_logs_15/qpik_ep15_seed12014.log
