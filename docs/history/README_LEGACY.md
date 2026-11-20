# MuJoCo benchmark v3.5: shared sensing and adaptation

This release adds an optional experiment that compares CT-SAC and Pure CPHT under original-state observations, an inferred state-history belief, and a belief with added geometry sensing. Existing experiments and checkpoints remain available.

Read [SENSING_EXPERIMENT.md](../SENSING_EXPERIMENT.md) for the input contracts, estimator, training supervision, fair comparison protocol, and server commands. [CHANGELOG_SENSING.md](CHANGELOG_SENSING.md) lists the changes. [VALIDATION_v3_5.md](VALIDATION_v3_5.md) records the executed checks.

Inspect the new plan without training:

```bash
CASE=both STAGE=prepare DRY_RUN=1 bash scripts/run_sensing.sh
```

Compare both methods using only the original instrumented state:

```bash
export CASE=both
export INFORMATION_MODES=state
export METHODS=ct_sac,pure_cpht
export SEEDS=0,1,2
export OPTIONS=cases/sensing.json

STAGE=prepare DRY_RUN=0 bash scripts/run_sensing.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_sensing.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 bash scripts/run_sensing.sh
```

Set `INFORMATION_MODES=state,state_belief,sensed_belief` to run the full information ablation. Each mode requires its own trained checkpoint. Set `METHODS` to a subset to control cost. The commands above are server commands; this release contains only short smoke checks.

The original-state mode has 31 channels, including exact simulated q/v and aggregate constraint-force ports. CPHT infers contact terrain from this permitted history and public robot geometry. The two belief modes have 60 channels. New presets fit CPHT's contact closure from aggregate force measurements and causal scans. Instrumented acceleration labels remain part of shared mechanics calibration. The guide explains the roles of memory, physical priors, and supervision in the comparison.

Run a short headless sensor precheck without training:

```bash
MUJOCO_GL=egl python preview_sensing.py --case J \
  --options cases/sensing_smoke.json --config cases/physical_first_smoke_task.json \
  --information sensed_belief --seconds 0.06 --video --out prechecks/sensing
```

The bundled `prechecks/sensing_v3_5/` shows sensor timing and rendering with zero actions. It provides no locomotion result.

The earlier Pure CPHT setup remains documented in [README_v3_4.md](README_v3_4.md) and [PURE_CPHT.md](../PURE_CPHT.md). Earlier physical-first and F/J value-solver experiments retain their versioned guides and launchers. Pure CPHT denotes experimental mechanical pH feedback/MPC; the manuscript's HJB/2PBVP value iteration is a separate implementation.
