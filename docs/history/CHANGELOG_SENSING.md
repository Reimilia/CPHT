# v3.5: matched state and sensing experiments

- Added opt-in `state`, `state_belief`, and `sensed_belief` modes.
- Preserved earlier cases, launchers, observation defaults, and checkpoint interfaces. The v3.4 guide remains in `README_v3_4.md`.
- Added a common causal geometry/friction estimator with acquisition timestamps, delay/dropout handling, validity flags, and uncertainty diagnostics.
- Enabled Pure CPHT to run from original-state-derived terrain estimates, with no terrain probe requirement in the new state modes.
- Added aggregate-port contact fitting to the new presets, without per-contact simulator supervision. Earlier point-label fitting remains available.
- Added hidden within-episode friction changes, optional randomized timing/scale, and frozen-checkpoint sensor/environment test overrides.
- Added algorithm/mode subset selection, strict paired comparison checks, sensor traces/plots, and a short headless preview.
- Added numerical, sensor-isolation, and train/checkpoint/evaluation integration tests. Full training remains a server task.

See `SENSING_EXPERIMENT.md` for observation dimensions, supervision scope, fairness, algorithms, commands, and limits.
