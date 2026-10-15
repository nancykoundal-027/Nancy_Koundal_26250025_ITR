# HEAL Headless IK Evaluation

Episodes per method: 15
Base seed: 12000
The same cube offsets are used across methods for each episode.
Wall runtime includes setup and computation, but excludes rendering and real-time sleeping.

| Method | Successes | Success rate | Lift verified | Place verified | Mean wall runtime (s) |
|---|---:|---:|---:|---:|---:|
| Mink | 15/15 | 100.0% | 15/15 | 15/15 | 3.3936 |
| DLS-IK | 15/15 | 100.0% | 15/15 | 15/15 | 3.0536 |
| QP-IK | 11/15 | 73.3% | 11/15 | 11/15 | 4.6526 |

## Interpretation note

These are system-level results from the implementations used here. Mink uses joint-space interpolation between IK configurations, while DLS and QP use Cartesian velocity tracking, so timing and trajectory comparisons are affected by controller architecture.

Do not treat a one-episode pilot as statistical evidence. Run 25 episodes per method for the main evaluation.
