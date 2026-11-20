# Validation of v3.1: finite course and speed tracking

Checks ran on 27 October 2026. No full calibration, experiment training, or costly
training run was launched. Short simulator checks used synthetic, untrained
checkpoint fixtures. Neural unit tests used tiny synthetic batches, including
single optimizer updates to check the update equations.

## Executed checks

- The full unit suite passed: **62 tests, zero failures, zero skips**, in 18.696 s.
  Run it with `MUJOCO_GL=egl python -m unittest discover -s tests -v`.
- Path-cost tests cover target-speed tracking, physical-time speed error,
  one-time goal/failure events, and the absorbing early-fall cost.
- Simulator tests check a goal reached inside an action hold, the exact measured
  terminal return, finite-horizon continuation, and deadline failure.
- All eight path-task adapters completed tiny numerical update checks. These
  include the source-style CT terminal boundary, requested versus elapsed hold,
  finite twin-critic recovery, and the existing contraction/integration checks.
- The learned live-transition ensemble test checks that the implementation
  averages joint survival/transition kernels. Replay and information-map tests
  retain the commanded hold separately from measured elapsed time.
- F and J integrated evaluation checks compare numerical task metrics with and
  without video capture. Their metrics agree exactly on the paired test
  trajectories. Each generated MP4 decodes, and each speed/progress PNG exists.
  The checks include the terminal display card and the final trajectory point.
- Separate 0.2 s F/J synthetic-checkpoint CLI rollouts produced headless MP4s,
  evaluation JSON, and speed/progress PNGs. Fixed and irregular clocks were
  exercised during development. The final fixed-clock previews were inspected.
- Legacy schema-2 checkpoint fixtures still render native and position-only F
  episodes under both clocks. These are compatibility fixtures, rather than
  the user's trained weights.
- Validation-only checkpoint selection, selected-checkpoint hash checks,
  completion-first aggregation, and undefined successful-completion statistics
  passed their tests.
- Algorithm selection tests cover named subsets, preset validation,
  deduplication, and shared preparation. F/J wrapper dry runs selected exactly
  `ct_sac,ph_sac` and retained that subset during later evaluation without a
  repeated `METHODS` argument. No jobs were launched by these dry runs.
- All supplied plans passed configuration and identity checks: 48 original-family
  preparation jobs, 384 original-family training jobs, 24 contact preparation
  jobs, 192 contact training jobs, 24 path preparation jobs, and 72 core-method
  path training jobs. These files are job descriptions, not completed runs.
- Python compilation, shell syntax checks, plan listing, and dependency-free
  command previews passed.

## Runtime used for these checks

The validation environment used Python 3.12, Torch 2.9.1+cpu, NumPy 2.3.5,
MuJoCo 3.17.0, dm-control 1.0.50, Gymnasium 1.4.0, imageio 2.38.1, and
imageio-ffmpeg 0.6.0. System EGL rendered without a display.
The source model can emit a `settotalmass` deprecation message with this MuJoCo
version. Its compiled mass remains the source value before an optional payload.

The declared server dependencies remain in `requirements.txt`. Its Torch pin
differs from this CPU validation wheel. CUDA execution and that exact server
stack were not tested here. Keep the actual package versions with each result;
evaluation JSON records them.

## Checks that remain for a compute server

Before a large array, use one small end-to-end preparation, train, resume, and
evaluation budget on the intended Torch/CUDA stack. Shared preparation requires
goal examples in both episode partitions and stops if coverage is missing.
The full preparation/training path and long-run policy quality were not measured
in this revision. Synthetic fixture output supplies no trained performance
result or guarantee of course completion.

Use independent training seeds to estimate policy variability. Use a physics-step
refinement study before reporting trajectory, slip, or work accuracy. The
contact audit performs endpoint re-solves on copied data and reports that extra
simulation work separately. Neither the task reward nor fitted-operator
contraction is a physical safety or policy-optimality certificate.

The earlier environment-only report is preserved as `VALIDATION_v3.md`.
Historical environment MP4s remain in `prechecks/contact/`.
