# Changes in v3.1

- Add a finite horizontal COM course, constant-speed tracking cost, explicit goal/fall/deadline events, and an absorbing early-failure cost.
- Add a declared noisy remaining-distance sensor, finite deadline information, and controlled calibration resets for terminal coverage.
- Separate commanded hold from elapsed time when an episode stops within a hold.
- Add a learned survival head and average ensemble joint live-transition kernels before conditional normalization.
- Use a measured terminal boundary for path CT adapters; retain original native-task targets.
- Integrate actor/observer-only checkpoint evaluation with JSON, physical-time MP4, terminal display cards, and PNG task plots.
- Bundle the earlier checkpoint renderer and support direct v2/v3 checkpoint replay without config migration.
- Add explicit method subsets and core/CT/solver/all presets; default to the three-method core instead of all eight.
- Add a path launcher, course/speed/deadline overrides, shared preparation, and per-episode training metrics.
- Add validation-only checkpoint selection, selected-checkpoint evaluation, stale-hash guards, path metric aggregation, and comparison figures.
- Move new configurations to schema 4. Earlier objective checkpoints retain their original semantics.
