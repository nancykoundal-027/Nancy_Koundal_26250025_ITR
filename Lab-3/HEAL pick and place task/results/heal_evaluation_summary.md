# HEAL Evaluation Summary

Success rates from the recorded baseline and wide stress tests.
The stress test varied initial cube x/y position by ±12 cm; grasp targets were shifted with the cube, so it is not a fixed-target test.

| Test | Method | Successes | Episodes | Success rate |
|---|---|---:|---:|---:|
| Baseline (±2 cm) | Mink | 25 | 25 | 100.0% |
| Baseline (±2 cm) | DLS-IK | 25 | 25 | 100.0% |
| Baseline (±2 cm) | QP-IK | 25 | 25 | 100.0% |
| Wide stress (±12 cm) | Mink | 15 | 15 | 100.0% |
| Wide stress (±12 cm) | DLS-IK | 15 | 15 | 100.0% |
| Wide stress (±12 cm) | QP-IK | 11 | 15 | 73.3% |

## Interpretation notes

- The baseline contains 25 episodes per method.
- The wide stress test contains 15 episodes per method.
- These tests vary starting positions but shift the grasp targets with the cube; results therefore do not measure fixed-target robustness.
- Wall runtime includes script setup and computation, and excludes viewer rendering and real-time sleeps.
- Mink uses joint-space interpolation between IK configurations, while DLS and QP use Cartesian velocity tracking. Runtime comparisons therefore reflect the complete implementations, not only solver speed.
- Successful-placement accuracy excludes failed episodes.
