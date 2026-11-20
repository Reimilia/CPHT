# v3.5 verification

Executed on 2026-11-01. No full training was launched.

## Runtime

- Python 3.12; Torch 2.5.1+cpu.
- MuJoCo 3.17.0; dm-control 1.0.50; Gymnasium 1.4.0.
- NumPy 2.3.5; SciPy 1.17.0; imageio-ffmpeg 0.6.0.
- Headless renderer: EGL.

## Checks

`MUJOCO_GL=egl python -m unittest discover -s tests -v`

Result: **100 tests passed, zero failures, zero skips**, in 81.212 seconds. The log is `prechecks/sensing_v3_5/tests_final.log`.

The suite includes seven new test methods. They verify polynomial/curvature recovery and coordinate transport, friction evidence gates, independent MuJoCo point Jacobians, state-only exclusion of probe/map input, sensor latency/clock accounting, hidden friction changes, and executable train/checkpoint/evaluation paths for both methods in all three modes.

The integration test uses 64 shared calibration decisions and four policy decisions per method/mode. It checks 31/60/60 observation dimensions, aggregate-port contact fitting without point-contact labels, frozen evaluation parameters, F-to-J transfer on an irregular clock, sensor-test overrides, strict comparison validation, and equal Pure CPHT motion/model results in `state` and `state_belief`.

The unchanged legacy tests also pass. Default legacy calibration/run identities are compared with v3.4 in `prechecks/sensing_v3_5/legacy_identity_check.json`.

The new shell runner passed syntax checks and dry runs for preparation, selected-method training, evaluation, and sensor-test overrides. Dry runs launch no simulator or training.

## Headless sensor precheck

Command:

```bash
MUJOCO_GL=egl python preview_sensing.py --case J \
  --options cases/sensing_smoke.json --config cases/physical_first_smoke_task.json \
  --information sensed_belief --seconds 0.06 --video --out prechecks/sensing_v3_5
```

The simulator ended this short check at 0.064 s, on its control boundary. Seventeen geometry frames were acquired, sixteen were delivered, and one remained delayed. Hidden friction scales changed at 0.02 s and 0.05 s. The script wrote MP4, PNG, and JSON. An independent run without recording produced exactly identical sensor/audit traces and task metrics.

These zero-action checks verify sensor plumbing, intervention timing, causal feature dimensions, and rendering. The interval contains one informative sliding contact. It cannot validate general friction recovery or uncertainty coverage. The synthetic curved-surface test supplies a separate numerical check of curvature recovery.

## Scope

Long-horizon locomotion, sample efficiency, robustness to unseen courses/materials, statistical uncertainty calibration, and any control advantage require the server experiments in `SENSING_EXPERIMENT.md`.

The original state includes exact simulated aggregate constraint forces. Mechanical calibration still uses instrumented acceleration. The new contact fit uses aggregate ports and causal scans; it receives no per-contact point/normal/gap/force or friction labels. CPHT has a known geometry/structured model and carries controller memory. CT-SAC's `state` actor is feedforward. The `state_belief` control exposes the same inferred history summary to CT-SAC. These differences define the interpretation of a fair deployed-information comparison.
