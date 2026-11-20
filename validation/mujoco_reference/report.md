# MuJoCo reference comparison

No training or parameter refitting was run. Each row is one paired episode. F fixes the physical regime/course; J draws the regime/course per episode. These runs do not impose an additional within-episode environment change.

| Case | Clock | Method | End | Distance (m) | Progress | Speed RMSE (m/s) | Saturation | Wall (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| F | fixed | learned_feedback | timeout | -2.567 | 0.0% | 3.224 | 34.6% | 17.4 |
| F | fixed | mujoco_ls | timeout | -2.750 | 0.0% | 3.235 | 35.3% | 10.8 |
| F | fixed | mujoco_fd_ls | timeout | -0.940 | 0.0% | 3.085 | 33.7% | 17.3 |
| F | fixed | learned_mpc | timeout | 4.392 | 22.0% | 2.652 | 48.4% | 43.2 |
| F | fixed | mujoco_mpc | timeout | 7.198 | 36.0% | 2.417 | 42.1% | 24.8 |
| F | irregular | learned_feedback | timeout | -1.505 | 0.0% | 3.132 | 34.6% | 31.9 |
| F | irregular | mujoco_ls | timeout | -1.265 | 0.0% | 3.112 | 34.9% | 15.9 |
| F | irregular | mujoco_fd_ls | timeout | -0.011 | 0.0% | 3.008 | 34.4% | 29.3 |
| F | irregular | learned_mpc | timeout | 2.741 | 13.7% | 2.778 | 39.4% | 53.0 |
| F | irregular | mujoco_mpc | timeout | 4.578 | 22.9% | 2.641 | 44.3% | 28.6 |
| J | fixed | learned_feedback | timeout | 1.124 | 5.6% | 2.917 | 31.0% | 17.9 |
| J | fixed | mujoco_ls | timeout | 1.019 | 5.1% | 2.925 | 31.5% | 10.4 |
| J | fixed | mujoco_fd_ls | timeout | -0.484 | 0.0% | 3.047 | 35.5% | 17.6 |
| J | fixed | learned_mpc | timeout | 3.201 | 16.0% | 2.767 | 46.8% | 45.3 |
| J | fixed | mujoco_mpc | timeout | 13.255 | 66.3% | 1.997 | 42.9% | 25.8 |
| J | irregular | learned_feedback | timeout | 1.590 | 7.9% | 2.881 | 33.0% | 28.8 |
| J | irregular | mujoco_ls | timeout | 0.916 | 4.6% | 2.937 | 32.1% | 14.0 |
| J | irregular | mujoco_fd_ls | timeout | 0.057 | 0.3% | 3.002 | 34.9% | 30.0 |
| J | irregular | learned_mpc | timeout | 1.727 | 8.6% | 2.893 | 43.8% | 54.6 |
| J | irregular | mujoco_mpc | timeout | 6.423 | 32.1% | 2.500 | 37.0% | 28.9 |

The MuJoCo arms use privileged model access. Their outcomes diagnose the selected controller and its finite gait/search space. A failure is not a proof of terrain infeasibility. Learned MPC and MuJoCo MPC share saved initial gait, parameter bounds, search count, cost weights, and control horizon; they also differ in feedback mechanics, contact model, numerical step, and scan-coverage cost (native: torso; learned: capsule points). This is a practical reference comparison, not a one-factor model ablation.

Training counters remain unchanged. Wall times include Python feedback and observation overhead and are not real-time guarantees.
