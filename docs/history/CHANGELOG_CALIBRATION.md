# Changes in v3.2.1

- Fix preparation failures caused by missing goal episodes in a train/validation partition.
- Collect bounded physical goal probes per known task inside the existing calibration decision budget.
- Split whole episodes within task/outcome strata, with explicit handling for rare classes.
- Preserve the coverage guard and check each known target speed.
- Save acquired.pt and calibration_acquisition.json before fitting. Retry fitting from saved acquisition under the same contract.
- Add progress output and detailed probe/outcome coverage diagnostics.
- Give the revised data protocol a new calibration/run identity.
- Add a data-only acquisition checker and a backup installer for existing path-aware F/J projects.
- Add end-to-end tiny preparation, retry, failed-probe, multiple-speed, split, and installer regression checks.
- Keep terrain construction, policy measurements, running objective, and RL update equations unchanged.
