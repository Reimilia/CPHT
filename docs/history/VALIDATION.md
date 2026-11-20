# Validation of v3.4: Pure CPHT

Checks ran on 31 October 2026. The final full regression suite passed **93 tests, zero failures, zero skips**, in 70.729 s. The executed command was unittest discovery over tests/. Its log is prechecks/pure_cpht/unit_tests.txt. No costly training or full-budget calibration was launched.

The checks establish numerical behavior and runnable interfaces. They provide no full-course locomotion ranking or learned-gait generalization result.

## Mechanical identification and phase-space consistency

The new Torch implementation was compared with independent MuJoCo evaluations at sampled configurations and velocities. Tests check mass matrices, bias/passive forces, kinetic/potential energy, and the 32-column inverse-dynamics regressor with tolerances around 1e-11 to 1e-12.

An independent automatic-differentiation/JVP check verifies that the acceleration implementation induces the canonical pH momentum equation. It also verifies the continuous power identity and a case with positive motor work. Positive mass/inertia/armature parameters preserve a positive mass matrix; the six motors retain three zero floating-root rows.

Synthetic perturbations change individual link parameters beyond the original three global scales. The new calibration fits held-out force equations to relative error below 1e-5 in that fixture. This checks the supported parameter family. It supplies no result for unknown geometry or arbitrary latent observation encoders.

The F/J command-line smoke calibrations used 64 control decisions each. Their held-out mechanical force RMSE was about 8.5e-12 and 1.0e-12, respectively, with exact simulator labels and a strongly informed public geometry/prior.

## Contact prediction and force observation

Tests check positive normal forces, Coulomb-cone feasibility, zero airborne contact force, and reduction of current force-port error by the causal observer. Motion planning is executed with future MuJoCo forward queries forbidden. The measured terrain-probe layout is checked against the existing observation source.

The approximate contact model has substantial held-out error. In the 64-decision smoke data, uncorrected generalized contact-port relative RMSE was about 0.84 in F and 0.32 in J. These values reflect a small data set, noisy scans, endpoint contact approximation, effective material parameters, and finite solver error. Mechanical identification accuracy remains a separate result.

The current measured force port can reduce local prediction error through the observer. This correction has finite memory and supplies no universal contact or transfer guarantee. Longer open-loop rollouts can amplify remaining error.

## Feedback and action holds

The bounded six-input quadratic solver was compared against independent SciPy L-BFGS-B solutions. The float64 random fixtures agreed in objective to approximately 1e-14, with KKT residual near machine precision. Runtime control uses float32 prediction quantities and reports its projected residual.

Tests verify one held command per prediction interval, including a longer measured initial hold and finer internal integration substeps. The controller advances its reference phase using actual elapsed physical time. Later predicted holds use the declared nominal planning clock; future sampled times remain unavailable.

One initial default planning solve took about 0.8 s on the shared CPU with one Torch thread. This is one timing sample. Feedback between plans is cheaper. CUDA latency, CUDA recovery, and real-time feasibility remain untested.

## Training, recovery, and evaluation

The new command-line launcher completed dry prepare/train/eval stages and short real stages for F and J, with Pure CPHT and a sensor-matched CT-SAC control. Each run used 64 calibration decisions and four subsequent decisions. Pure CPHT performed physical-model refits and zero actor/critic updates.

There were 13 completed evaluation reports and 13 verified headless MP4s, covering fixed/irregular clocks, saved-gait feedback, receding-horizon feedback, and F-to-J step transfer. These task episodes lasted 0.08 s. The comparison script accepted the paired three-method reports in each case.

Regression tests also verify bitwise CPU recovery of the learned physical parameters and gait warm start after an episode-boundary checkpoint, zero SAC learner construction in Pure CPHT training, unchanged metrics/prediction errors with recording, and frozen deployment models.

Sixteen existing F/J jobs were compared against v3.3 configuration functions. Their default settings, calibration identities, and run identities were unchanged. Earlier checkpoint interfaces and legacy value-solver tests remain covered by the passing suite.

## One-second feedback feasibility check

The bundled model JSON uses 64 calibration decisions and a gait warm start from four control interactions. The precheck uses the ordinary 12 s task deadline and stops its recording after one second. It performs no model, actor, or critic updates.

On the initial flat pad of fixed course 1, the saved-reference feedback controller advanced about **0.799 m** and ended near **1.832 m/s** without falling. Its one-second speed RMSE was about **2.291 m/s**, including acceleration from rest. The path task remained unfinished.

This is an actuation feasibility check under a seeded periodic reference. It establishes neither a learned stable limit cycle nor a gain caused by physical learning. The zero-seed and unlearned-model controls, complete path, changing terrain, and independent training seeds remain necessary experiments.

Run preview_pure_cpht.py with prechecks/pure_cpht/feedback_precheck_model.json to reproduce the video. Its one-second recording and summary are included. This precheck is separate from the 0.08 s interface smoke runs.

## Runtime

Checks used Python 3.12, Torch 2.9.1+cpu, NumPy 2.3.5, SciPy 1.17.0, MuJoCo 3.17.0, dm-control 1.0.50, EGL, and FFmpeg. Historical validation is retained in VALIDATION_v3_3.md and earlier versioned notes.

The guide PURE_CPHT.md specifies the physical family, sensors, bounded feedback, numerical approximations, task protocol, and unresolved learning claims.
