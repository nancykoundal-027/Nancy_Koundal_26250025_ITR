# ITR Lab 3: Franka Panda Pick-and-Place Using Inverse Kinematics

## 1. Project Overview

This project implements and compares three inverse-kinematics (IK) methods for pick-and-place manipulation using a Franka Panda robot in MuJoCo.

The robot interacts with a simulated table, cube and tray. The task involves approaching and grasping the cube, lifting it, transporting it to the tray, placing it and retreating.

The project also includes robot workspace analysis, randomized episode testing, controller performance metrics and joint-constraint evaluation.

## 2. IK Methods

1. **Mink IK:** Uses the Mink framework and its DAQP-based IK solver.
2. **Damped Least-Squares (DLS) IK:** Calculates joint updates using a damped least-squares method.
3. **Custom QP IK:** Uses a custom quadratic-programming formulation with joint-position and per-step motion constraints.

## 3. Repository Structure

- `robot_descriptions/franka_emika_panda/`: robot model, scene and mesh assets.
- `scripts/`: workspace analysis, validation and IK benchmark scripts.
- `results/`: episode logs, workspace summaries and CSV comparison tables.
- `plots/`: workspace, performance and constraint-comparison plots.
- `failure_screenshots/`: representative simulation failure screenshots.
- `report/`: final written report.
- `video/`: three-minute IK demonstration video.

## 4. Software Requirements

The project uses Python, MuJoCo, Mink, NumPy, SciPy, Matplotlib and DAQP.

Activate the existing Conda environment with:

    conda activate ros_env

Run commands from the `ITR_LAB3_FRANKA` project root.

## 5. Workspace Analysis

Run the workspace analysis using:

    python scripts/workspace_analysis.py

Workspace results and plots are saved under `results/` and `plots/`. Key plots include `task_workspace.png`, `task_workspace_xy.png`, `task_vs_dexterous_workspace.png` and `yaw_workspace.png`.

Workspace sampling is approximate. It does not by itself guarantee collision-free motion or successful physical pick-and-place.

## 6. Running the IK Benchmarks

Run the 25-episode benchmarks with seed 0:

Mink:

    python scripts/mink_speedcap_test.py --episodes 25 --seed 0

Damped Least-Squares IK:

    python scripts/dls_speedcap_test.py --episodes 25 --seed 0

Custom QP IK:

    python scripts/qp_speedcap_test.py --episodes 25 --seed 0

Each script saves its results and performance plot. Check the printed output paths after running each test.

## 7. Recorded Benchmark Results

The current recorded 25-episode speed-cap results are:

| Metric | Mink | DLS | Custom QP |
|---|---:|---:|---:|
| Successful episodes | 18/25 | 17/25 | 18/25 |
| Success rate | 72% | 68% | 72% |
| Mean episode time | 2.693 s | 2.321 s | 2.841 s |
| Mean IK iterations | 6906.3 | 6913.3 | 7556.1 |
| Episodes with joint-position violations | 4 | 2 | 1 |
| Episodes exceeding 0.45 rad/s | 7 | 4 | 9 |
| Peak measured joint speed | 1.173 rad/s | 1.283 rad/s | 4.240 rad/s |

All three methods used the same initial cube positions and yaw angles in these recorded runs.

DLS had the lowest mean episode time and the fewest episodes exceeding the 0.45 rad/s threshold. Mink and Custom QP achieved the highest success rate. However, all three methods recorded constraint violations; the results must not be described as fully compliant.

The commanded speed cap is not the same as measured joint speed. Actuator dynamics can cause the actual speed to exceed the command limit.

## 8. Comparison Plots and Logs

The final comparison plots are:

- `plots/final_ik_success_rate_speedcap.png`
- `plots/final_ik_mean_time_speedcap.png`
- `plots/final_ik_mean_iterations_speedcap.png`
- `plots/final_ik_peak_speed_speedcap.png`
- `plots/final_ik_constraint_violations_speedcap.png`

The summary table is `results/final_ik_comparison_speedcap.csv`.

The failure-analysis table is `results/final_ik_failure_analysis_speedcap.csv`.

Per-episode JSON logs are also available under `results/`.

## 9. Failure Analysis

Failed episodes should be examined using their logged failure reason, initial cube pose, phase errors and constraint measurements. Representative screenshots and explanations should be placed in `failure_screenshots/` and included in the final report.

The pick-and-place workspace validation scripts are diagnostics; target reachability alone does not establish successful grasping, collision-free transport or placement.

## 10. Table Design Justification

The table, tray and cube geometry is defined in `robot_descriptions/franka_emika_panda/pick_place_scene.xml`.

The report should document the actual dimensions and surface heights from this scene and justify the table and tray positions using the workspace plots and tested target poses. Sampled workspace estimates are approximate, not proof that every point is reachable.

## 11. Required Deliverables

1. Source code with setup and run instructions.
2. Workspace plots and table-design justification.
3. Per-episode logs and summary metrics.
4. Failure-case report with representative screenshots.
5. Comparative IK analysis and plots.
6. A three-minute video with approximately one minute each for Mink, DLS and Custom QP, showing the robot trajectories and pick-and-place behaviour.

Place the final report in `report/` and the demonstration video in `video/`.

## 12. Reproducibility

The benchmark commands use a common seed base to compare methods on matching randomized cube poses. Results can depend on software versions, model files and controller settings.

Retain the original robot model's license and attribution files when distributing its description and assets.

## Video Demonstration

[Watch the FRANKA Pick-and-Place IK Comparison on YouTube](https://youtu.be/JxW19l81Ryw)
