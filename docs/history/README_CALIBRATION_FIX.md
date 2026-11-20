# Fix for missing goal examples during preparation

The error at `fjbench/calibration.py` line 107 in the earlier bundle comes from
the calibration coverage check. A training or validation partition had no
successful goal episode. The collector relied on random reset choices, then
assigned every fifth episode to validation. This procedure could miss the goal
boundary or place all successful episodes in the training partition.
The earlier environment/solver smoke checks did not exercise full preparation.

## Install in the existing project

Extract the updated bundle. Use its project directly, or install the fix in the
existing project folder:

```bash
python /path/to/updated/TPHT_CT_RL_Benchmark/apply_calibration_fix.py \
  --target /path/to/TPHT_CT_RL_Benchmark_v2
```

The target can have any folder name. The installer checks for the path-aware
v3.1/v3.2 implementation, backs up changed files, and copies the calibration fix.
Backups go to `calibration_fix_backups/`. `--dry-run` lists the proposed files.
The installer adds `CALIBRATION_PATCH_VERSION.json` to record the patch.

From the patched target, rerun the same command:

```bash
STAGE=prepare DRY_RUN=0 bash scripts/run_path_speed.sh
```

Keep the selected case, method subset, configuration, training seed, and device settings used
for the intended experiment. The revised calibration protocol receives a new
calibration/run identity. Preparation writes its new shared artifact to that
identity's directory.

## Acquisition and splitting

The collector first seeks four measured goal episodes per known target speed.
These are separate physical rollouts with distinct reset seeds. It starts each
goal probe at

$$
x_0=x_g-\min(0.01L,\;0.25v_\star\Delta t_{\rm physics}),
\qquad v_0=v_\star.
$$

The source pose is sampled and settled through the existing reset procedure.
The existing calibration reset interface then sets the requested COM position
and forward speed. Uniform random bounded actions drive actual MuJoCo steps.
The goal, upright condition, measured holding time, reward, and termination all
come from the environment. Failed probes remain failed records.

Each task has at most 32 probe attempts, with at most four decisions per attempt.
All probe interactions count toward `calibration_decisions`; its default remains
30000. The rest of that budget uses the existing ordinary calibration reset law.
Policy training and evaluation keep their course-start reset and task objective.

The split groups episodes by known task and observed terminal outcome.
Groups with two to four episodes contribute one whole recorded episode to validation.
Larger groups retain the every-fifth convention within the group. Singletons
remain in training. An episode and its history stay in one partition.
Every known task must have a measured goal episode in both partitions before
fitting begins. This remains an admission check for the collected data.

The new configuration keys are `calibration_protocol=goal_balanced_v2`,
`path_calibration_goal_episodes=4`, `path_calibration_goal_attempts=32`, and
`path_calibration_probe_steps=4`. A custom goal target requires at least two
episodes. The per-task probe target and retry limit can be increased explicitly.
These settings change the calibration identity.

## Diagnostics and reuse

Preparation saves `acquired.pt` before admission and neural fitting.
`calibration_acquisition.json` contains collected goals, train/validation counts
per target speed, probe attempts, terminal causes, and acquisition completeness.
If a later fitting attempt fails, rerunning preparation reuses that measured
data after checking the configuration contract. A failed admission retains the
guard and writes the evidence before stopping.

The earlier failing version saved its acquisition only after successful fitting.
Its already-exited failing run has no acquisition cache to recover. The first
patched preparation collects a new dataset.

Use this small data-only check before full preparation:

```bash
python check_calibration_fj.py --case both --decisions 128
```

It collects 128 decisions per case, reports goal coverage, and performs zero
observer, operator, or policy updates. The output is a diagnostic dataset report;
it supplies no training-quality or course-completion result.

The patch was checked with short F/J acquisitions for both terrain backends,
multiple target speeds, bounded failed probes, and tiny preparation/retry tests.
The coverage check requires measured examples. It supplies no global exploration,
model accuracy, or policy-optimality guarantee.
