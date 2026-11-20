# MuJoCo benchmark v3.2.1: CT terrain and reliable F/J preparation

This revision fixes the missing-goal coverage error during `STAGE=prepare`.
It collects bounded physical goal probes within the calibration budget, keeps
whole episodes in task/outcome strata, and saves acquisition before fitting.
[README_CALIBRATION_FIX.md](README_CALIBRATION_FIX.md) gives the diagnosis,
small data-only check, and an installer for an existing project folder.

This revision aligns the physical terrain with
[sebale16/ct-rl, commit 297ecbed](https://github.com/sebale16/ct-rl/blob/297ecbedb1b7510439f85927aa4b70d893f681e8/docs/cheetah_terrain.md).
It retains the finite `path_speed` task, current numerical sensors, algorithm subsets,
and checkpoint evaluation with MP4/PNG output.
F uses fixed course seed 1. J draws a fresh full-difficulty course per episode.
Hills, friction patches, combined terrain, slopes, and steps have separate presets.
[TERRAIN_ALIGNMENT.md](../TERRAIN_ALIGNMENT.md) defines the physical match and the executed source checks.
The earlier mesh/material environments remain available through `cases/path_speed_mesh.json`
and the `contact_*.json` presets.

The default path task has a 20 m forward course, a 3 m/s reference speed, and a 12 s deadline.
The source reset samples and settles the pose; the path task then clears velocity. Speed error includes acceleration from rest.
The reward penalizes speed error and control effort. It pays a goal bonus and charges failure costs.
Overspeed receives a tracking penalty.
Every method receives the same noisy remaining-distance sensor and the same exogenous action-hold rule.

The finite task changes the objective, observation size, and terminal model.
Train new checkpoints for this task. Existing v2/v3 checkpoints can still be evaluated under their saved tasks.
[PATH_SPEED_PROTOCOL.md](../PATH_SPEED_PROTOCOL.md) defines the objective, data, terminal updates, and metrics.
Historical instructions remain in `README_v3_2.md`, `README_v3_1.md`, `README_v3.md`, `README_v2.md`, and `README_v1.md`.

## Select the physical terrain

```bash
# Preview two selected algorithms on fixed and resampled hills.
TERRAIN=hills CASE=both METHODS=ct_sac,ph_sac SEEDS=0 STAGE=train \
  bash scripts/run_terrain.sh
```

Use `TERRAIN=friction` for flat friction patches or `TERRAIN=terrain` for hills plus friction.
`TERRAIN=slopes` and `TERRAIN=steps` select the source's optional elevation laws plus friction patches.
The terrain launcher keeps each variant's saved plans in `plans/ct_terrain/<variant>/`.
It delegates the same prepare/train/eval/select stages used below and defaults to dry run.
The existing `run_path_speed.sh` now uses combined CT terrain by default.

## Select a small experiment before launching work

Run these commands from the extracted `TPHT_CT_RL_Benchmark` folder:

```bash
# List exact method names and presets. This launches no work.
python make_fj_plan.py --list-algorithms

# Inspect two methods for both F and J, one training seed each.
CASE=both METHODS=ct_sac,ph_sac SEEDS=0 STAGE=train \
  bash scripts/run_path_speed.sh
```

The wrapper defaults to `DRY_RUN=1`.
The command above creates two preparation jobs and four training jobs, then prints their configurations.
The `core` preset is `ct_sac,interval_euler,ph_sac`.
`ALGORITHM_SET=solver` selects `model_sac,interval_euler,ph_sac,ph_direct`.
`ALGORITHM_SET=all` explicitly selects all eight methods.
An explicit `METHODS` list overrides the preset. Names are validated and duplicates are removed.

Choose a custom course or deadline before preparation:

```bash
CASE=J METHODS=interval_sac,ph_sac PATH_LENGTH=30 TARGET_SPEED=3 DEADLINE=16 \
  STAGE=train bash scripts/run_path_speed.sh
```

Use `CONFIG=your_overrides.json` with `run_terrain.sh` for neural/solver budgets or sensor settings.
That launcher retains its selected physical preset when it merges the custom JSON.
With `run_path_speed.sh`, `CONFIG` selects the complete preset; include `contact_backend=ct_terrain`
in a custom contact configuration to select the matched backend. Use `contact_fixed_seed=1` for course 1.
The path wrapper always selects `control_task=path_speed`.
`PATH_LENGTH`, `TARGET_SPEED`, and `DEADLINE` override the corresponding JSON settings.
A different task or sensor receives a different calibration/run identity.

## Prepare and train only the selected algorithms on the server

Activate the server's existing Python/Torch environment and install the project dependencies as needed.
Keep the same Torch/MuJoCo stack when replaying an existing checkpoint.
Use the same selection and task parameters at each stage:

```bash
export CASE=F
export METHODS=ct_sac,ph_sac
export SEEDS=0
export DEVICE=cuda

STAGE=prepare DRY_RUN=0 bash scripts/run_path_speed.sh
STAGE=train   DRY_RUN=0 bash scripts/run_path_speed.sh
```

Preparation is shared once per case/seed, including observer and operator fitting.
Training launches only the selected methods.
Set `CASE=J` for the sampled contact family, or `CASE=both` for both cases.
`EXPERIMENTS=contact` is the default. The CT terrain backend uses this experiment.
For the earlier flat-ground, position-only, static, or wind experiments, set
`CONFIG=cases/path_speed_mesh.json` and the corresponding `EXPERIMENTS` value.

The decision budget remains 1,000,000 including 30,000 calibration decisions.
Resume at an episode boundary with `RESUME=1`.
`episodes.jsonl` now records course progress, terminal cause, completion, and speed metrics during training.
A budget cut inside an episode is marked incomplete.

No costly training was launched while preparing this revision.

## Evaluate checkpoints and record videos in the same rollout

```bash
# Final checkpoint, paired fixed/irregular clocks, five episodes each,
# one headless video per clock, plus task JSON and speed/progress PNGs.
CASE=F METHODS=ct_sac,ph_sac SEEDS=0 STAGE=eval DRY_RUN=0 \
  DEVICE=cpu EPISODES=5 VIDEOS=1 bash scripts/run_path_speed.sh

# J: fresh courses from the source law, then the separate layout seed stream.
CASE=J METHODS=ct_sac,ph_sac SEEDS=0 STAGE=eval DRY_RUN=0 \
  DEVICE=cpu EPISODES=5 VIDEOS=1 \
  COHORTS=iid,layout bash scripts/run_path_speed.sh
```

Metrics and plots go to each run's `evaluations/` directory. MP4s and video sidecars go to `videos/` in that run.
With the terrain launcher, retain the same `TERRAIN` value at evaluation:

```bash
TERRAIN=hills CASE=J STAGE=eval DRY_RUN=0 EPISODES=5 VIDEOS=1 COHORTS=iid,layout \
  bash scripts/run_terrain.sh
```

Videos show speed, target, remaining distance, and a labelled terminal state: goal, fall, or timeout.
Frames follow physical time during irregular holds.
The default one-second terminal display hold is labelled separately and excluded from physical-time metrics.
Use the direct CLI's `--end-card-seconds 0` to disable it.
Set `VIDEOS=0` for metrics/plots alone and `CLOCKS=fixed` for one clock.
Evaluation and selection reuse the saved training plan. An explicit `METHODS` list can narrow that saved set.
Task overrides apply to preparation/training; deployed checkpoints retain their saved task contract.
The MP4, JSON, and PNG use the same evaluated trajectory; recording supplies no extra actor observation.

A direct job command is:

```bash
python eval_fj.py --case F --experiment contact --config cases/path_speed.json \
  --method ph_sac --seed 0 --checkpoint final --episodes 5 \
  --video-episodes 1 --clocks fixed,irregular --gl egl --device cpu
```

Use `--video-seconds 3` to save only a short clip while completing numerical evaluation.
Use `--overwrite-videos` for an intentional rerender.
`--gl osmesa` uses a system OSMesa installation when EGL is unavailable.
`requirements-video.txt` contains the video-specific Python dependencies.

## Select a checkpoint on validation data

Evaluate candidate checkpoints on the validation split first:

```bash
CASE=F METHODS=ct_sac,ph_sac STAGE=eval SPLIT=validation CHECKPOINT_TAG=500000 \
  EPISODES=5 VIDEOS=0 DRY_RUN=0 bash scripts/run_path_speed.sh
CASE=F METHODS=ct_sac,ph_sac STAGE=eval SPLIT=validation CHECKPOINT_TAG=final \
  EPISODES=5 VIDEOS=0 DRY_RUN=0 bash scripts/run_path_speed.sh
CASE=F METHODS=ct_sac,ph_sac STAGE=select EPISODES=5 DRY_RUN=0 \
  bash scripts/run_path_speed.sh

# Apply the selected checkpoint to the held-out test set.
CASE=F METHODS=ct_sac,ph_sac STAGE=eval CHECKPOINT_TAG=selected \
  EPISODES=5 VIDEOS=1 DRY_RUN=0 bash scripts/run_path_speed.sh
```

Selection averages the requested validation clocks and ranks completion rate first, then relative speed RMSE.
It reads validation results only and writes `selected_checkpoint.json` without modifying weights.
Checkpoint hashes protect selection and aggregation against stale results.

## Aggregate and compare only completed runs

```bash
python aggregate_fj.py --plan plans/path_speed/F_eval.jsonl --root runs \
  --clock fixed --episodes 5 --checkpoint selected --reference ct_sac \
  --out reports/F_path_fixed.json

python plot_fj.py --evaluations runs/FJ/F/contact/*/seed*/evaluations/selected_test_iid_fixed_n5.json \
  --methods ct_sac,ph_sac --out reports/F_path_comparison.png
```

Completion rate is the primary path metric. Report speed RMSE and time in the target band alongside it.
Completion time is conditional on success and remains undefined when there are no completions.
Aggregation keeps independent training seeds as the uncertainty unit.
A one-seed run is a pilot check; repeated evaluation episodes alone do not replace independent training seeds.
The comparison plot accepts at most six selected methods and one case/cohort/clock at a time.

## Replay an existing v2/v3 checkpoint

```bash
python eval_fj.py --checkpoint-file /path/to/existing/final.pt \
  --episodes 2 --clocks fixed,irregular --video-episodes 1 --device cpu --gl egl
```

Checkpoint-file mode restores the saved objective and observations. It accepts no task/config override.
For an exact recorded v2 episode and checkpoint-hash check, the bundled `render_fj.py`
still accepts `--evaluation-json`; see `README_RENDER_FJ_V2.md`.
Older weights keep their earlier task semantics.

## Plans, Slurm, and prechecks

The supplied `plans/path_speed_prepare.jsonl` has 24 preparation jobs on combined CT terrain.
`plans/path_speed_train.jsonl` has 72 training jobs: two contact cases, twelve seeds, and the three core methods.
`plans/ct_terrain/` also has separate hills, friction, and terrain plans with the same counts.
These are alternatives; the launcher executes only the selected variant and methods.
Generate a smaller plan with `--config cases/path_speed_ct_hills.json --experiments contact --methods ct_sac,ph_sac --seeds 0`.
Reuse `scripts/slurm_fj.sh` with the selected plan and array range.

Environment-only sampling and previews remain available:

```bash
python sample_envs.py --case both --config cases/path_speed.json --count 1 --plot \
  --out prechecks/path_speed/environments.jsonl
python preview_envs.py --manifest prechecks/path_speed/environments.jsonl --limit 2 --seconds 2
```

Those controllers are unfitted prechecks. Historical MP4s in `prechecks/contact/` are the earlier v3 environment examples.
[VALIDATION.md](VALIDATION.md) records the executed numerical checks and their limits.
