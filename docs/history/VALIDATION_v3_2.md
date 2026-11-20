# Validation of v3.2: CT terrain alignment

Checks ran on 27 October 2026. No full calibration or experiment training was
launched. Short neural checks use synthetic batches and untrained checkpoint
fixtures. Environment previews use a fixed sine controller or a camera tour.

## Source comparison

The physical reference is sebale16/ct-rl commit
`297ecbedb1b7510439f85927aa4b70d893f681e8`, file
`environment/cheetah_terrain.py`, Git blob
`e95e50779d6389d2f1bc19b5f10f7fd8a518efec`.
The downloaded source blob was verified before import.

`verify_terrain_match.py` checked ten courses: hills, friction, terrain, slopes,
and steps at seeds 1 and 7. All sampled positions, heights, patch edges, and
friction arrays agreed exactly with the source. Checked compiled model arrays
also agreed exactly: masses, inertias, damping, motor gears, ground geometry,
collision flags, friction, solver parameters, colours, and heightfield data.
Gravity, integrator, and physics timestep matched on the validation stack.

Each course passed 997 vertical surface rays. The largest height error against
the reference profile was 5.96e-8 m. The full results are in
`prechecks/ct_terrain/upstream_match.json`. Independent numerical course fixtures
and their provenance are in `tests/fixtures/ct_terrain_reference.*`.

The comparison covers the physical course and robot. The benchmark retains its
current sensors, explicit episode-seed schedule, calibration resets, finite path
objective, and action-hold protocol. `TERRAIN_ALIGNMENT.md` defines these parts.

## Regression and integration checks

- **70 tests passed, zero failed, zero skipped**, in 23.779 s. The complete log is
  `prechecks/ct_terrain/unit_tests.txt`.
- Course tests cover exact source fixtures, fixed F seed 1, new J draws, paired
  episode seeds, the level start pad, RNG draw order, full difficulty, and
  rejection of unsupported material/curriculum overrides.
- Simulator tests check the five physical variants, surface interpolation, stock
  mass and actuator dimensions, and effective friction on a patch below 0.4.
- F/J synthetic-checkpoint evaluations passed under fixed and irregular clocks.
  Numerical path metrics agree exactly with and without video capture.
  Validation-split evaluation also passes for fixed F contact environments.
- Eight headless precheck videos decoded: six one-second F/J rollouts across the
  three main variants, plus two two-second combined-terrain camera tours.
  They contain 250 frames in total at 640 by 360 pixels and 25 fps.
  Sampled terrain figures and representative video frames were visually checked.
  Effective friction and contact-response audit errors were zero in these
  prechecks. `prechecks/ct_terrain/video_checks.json` records the video checks.
- Tiny update, finite-boundary, replay, solver, and legacy checkpoint-rendering
  tests from v3.1 remain part of the passing regression suite.
- F/J dry launches retained exactly `ct_sac,ph_sac` across train/eval/select and
  created one shared preparation job per case/seed. A custom neural JSON keeps the selected physical preset. No training jobs ran.
- All supplied F/J plans passed configuration and run/calibration identity
  checks. There are 1032 job descriptions across the historical and new plan
  files. The three terrain-specific plan pairs are alternatives, selected
  explicitly through `TERRAIN`.
- Python compilation, shell syntax, dependency-free plan generation, and
  launcher previews passed.

## Validation runtime and limits

The checks used Python 3.12, Torch 2.9.1+cpu, NumPy 2.3.5, SciPy 1.17.0,
MuJoCo 3.17.0, dm-control 1.0.50, Gymnasium 1.4.0, imageio 2.38.1, and
imageio-ffmpeg 0.6.0. EGL rendered without a display. Torch differs from the
server wheel declared in `requirements.txt`; CUDA execution was not tested here.
MuJoCo emits the source model's `settotalmass` deprecation message on this stack.
Its compiled cheetah mass remains 14 kg before any separately declared payload.

The source and adapter were compared with the same installed simulator stack.
No trained completion, speed-tracking quality, full calibration coverage,
long-run recovery, or training throughput result was measured in this revision.
Use a small end-to-end server preparation/train/resume/eval budget before a large
array. The calibration goal-coverage guard remains active. Use independent
training seeds for policy uncertainty and a timestep refinement study for
trajectory/contact accuracy.

Earlier reports remain in `VALIDATION_v3_1.md`, `VALIDATION_v3.md`,
`VALIDATION_v2.md`, and `VALIDATION_v1.md`.
