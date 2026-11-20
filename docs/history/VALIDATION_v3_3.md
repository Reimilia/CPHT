# Validation of v3.3: physical calibration first

Checks ran on 28 October 2026. The full regression suite passed **85 tests, zero failures, zero skips**, in 48.272 s. A further acquisition-retry regression passed in 6.233 s. After adding the XML identity check and CUDA RNG conversion, the CPU training/recovery/evaluation regression passed again in 12.739 s. There are 86 distinct tests in the delivered suite. Logs are in `prechecks/physical_first/`.

No full experiment training or full-budget physical calibration was launched. The command-line smoke checks used 64 calibration decisions per case and 16 policy updates per selected case/method. These short runs establish executable behavior. They provide no evidence that a trained cheetah completes the full path or that one method generalizes better.

## Physical model and measurements

The F and J smoke acquisitions each recorded 156 physics-step labels from 64 control decisions. Calibration used disjoint whole-episode training and validation folds. It learned three positive corrections around the known XML template. Both fits recovered the expected coefficients approximately `(1.25, 0.833333333333, 0.714285714286)`.

F held-out force RMSE was `8.45e-12`; J held-out force RMSE was `1.08e-12`, using exact instantaneous MuJoCo labels. The deliberately biased nominal template had held-out force RMSE about 16.74 in F and 15.07 in J. Translational and angular equation rows have their respective generalized-force units. These values reflect the exact, restricted calibration family and instrumented forces.

The regression tests verify a non-diagonal mass matrix, moving point-Jacobian construction, and a back-foot port with zero direct front-joint torque and nonzero front-joint acceleration response. An independent energy-gradient calculation checks the continuous physical power identity. Copied state/force queries, including physics-rate calibration hooks, preserve the simulator trajectory and warm start.

A planned fit failure preserved raw acquisition. Retrying with reacquisition forbidden completed the fit from the saved samples. Calibration control records and physics labels retain consistent episode membership.

## Training and evaluation behavior

The actual `scripts/run_physical_first.sh` launcher completed dry preparation, training, and evaluation for `ct_sac,cpht_sac` on F and J. It then completed the small real smoke stages with CPU updates, final and recovery checkpoints, six evaluation reports, and four EGL MP4 recordings. The extra two evaluations transferred the F-trained policies to J step courses under an irregular clock with frozen physical coefficients and policy weights.

The comparison script accepted the paired fixed-clock evaluations in each case and wrote JSON and PNG summaries. It checked common calibration IDs, actual training budgets, evaluation episode seeds, and course hashes. An MP4 frame and comparison plot were inspected. The test clips use a 0.08 s task deadline and a labelled one-second terminal display hold.

Separate tests verify terminal targets from measured interval returns, zero calls to the finite value inverse, bitwise CPU actor equality after episode-boundary recovery, unchanged evaluation metrics with recording, and frozen F-to-J transfer. All four optional method combinations execute their update rules in a small tensor test.

`prechecks/physical_first/cli_smoke_summary.json` records the command stages, coefficients, counters, and evaluation metrics. Contact/Jacobian examples and smoke comparison plots are also included. Smoke policy weights and replay files are excluded from the download.

## Legacy residual recovery

The numerical stress fixture reproduces failure of the old absolute gate: its recovery defect is `6.311e-8` with normalized defect `7.663e-17`. The scale-aware gate passes it. The fixture has discount 0.9999 and a linear-system 2-norm condition number about 10000. `finite_recovery.json` contains the exact diagnostics and zero-training declaration. Passing recovery accuracy does not imply convergence of the finite value iteration; those are separate reported fields.

The suite also verifies that a true recovery failure leaves actor/critic parameters unchanged. New numerical-only recovery settings preserve prior calibration identities. The installer was applied to a small existing-folder fixture: configuration bytes remained unchanged and both replaced source files had verified backups.

All 1032 existing plan entries were normalized to the new numerical fields and checked for configuration, calibration, and run identity. Six current CT terrain preview manifests received the same numerical defaults and updated config hashes; their physical environments are unchanged. Compilation and shell syntax checks passed.

## Runtime and limits

Checks used Python 3.12, Torch 2.9.1+cpu, NumPy 2.3.5, SciPy 1.17.0, MuJoCo 3.17.0, dm-control 1.0.50, and EGL. CUDA training/recovery and OSMesa rendering remain untested here.

This pilot uses exact q/v, instrumented force ports, and a three-parameter correction of one known cheetah. It evaluates a physical representation with conventional SAC/CT-SAC targets. General physical-Hamiltonian discovery, arbitrary robot configurations, learned force sensing, closed-loop passivity, the CPHT HJB/2PBVP theorem, and trained policy quality require separate experiments.

Historical validation is preserved in `VALIDATION_v3_2_1.md` and earlier versioned notes. The existing full-budget jobs remain unlaunched descriptions.
