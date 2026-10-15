# HEAL Headless IK Evaluation

Episodes per method: 25
Base seed: 2026
The same cube offsets are used across methods for each episode.
Wall runtime includes setup and computation, but excludes rendering and real-time sleeping.

| Method | Successes | Success rate | Lift verified | Place verified | Mean wall runtime (s) |
|---|---:|---:|---:|---:|---:|
| Mink | 25/25 | 100.0% | 25/25 | 25/25 | 3.3709 |
| DLS-IK | 25/25 | 100.0% | 25/25 | 25/25 | 3.0489 |
| QP-IK | 25/25 | 100.0% | 25/25 | 25/25 | 5.7481 |

## Interpretation note

These are system-level results from the implementations used here. Mink uses joint-space interpolation between IK configurations, while DLS and QP use Cartesian velocity tracking, so timing and trajectory comparisons are affected by controller architecture.

Do not treat a one-episode pilot as statistical evidence. Run 25 episodes per method for the main evaluation.
