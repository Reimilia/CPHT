# MuJoCo benchmark v3.3: physical calibration first

The new experiment learns three positive mass, stiffness, and damping corrections from measured cheetah dynamics. It freezes the physical pH model and trains SAC with momentum, energy, and power features. CT-SAC receives the same measured state and environmental force ports. This pipeline uses measured transitions and performs zero finite critic-residual inverse solves.

[PHYSICAL_FIRST.md](../PHYSICAL_FIRST.md) contains the installation steps, calibration equations, sensor contract, update rules, algorithm sketch, checkpoint evaluation, transfer tests, and scope limits. The method key `cpht_sac` selects this physical-first SAC pilot. It tests a calibrated representation; the manuscript's HJB/2PBVP value solver remains a separate experiment.

Run commands from the extracted `TPHT_CT_RL_Benchmark` folder. Activate your Torch environment and install `requirements-physical-first.txt` as needed. Start with a dry preview:

```bash
CASE=both METHODS=ct_sac,cpht_sac SEEDS=0 STAGE=prepare \
  bash scripts/run_physical_first.sh
```

The launcher defaults to `DRY_RUN=1`. To run on your server:

```bash
export CASE=both
export TERRAIN=terrain
export METHODS=ct_sac,cpht_sac
export SEEDS=0
export MUJOCO_GL=egl

STAGE=prepare DRY_RUN=0 bash scripts/run_physical_first.sh
STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_physical_first.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 \
  bash scripts/run_physical_first.sh
```

Calibration uses 2048 decisions per case/seed and zero policy updates. The default total budget is 1,000,000 decisions, including calibration. Choose one method with `METHODS=cpht_sac`. Optional controls are `sac` and `ct_sac_ph`; they separate the physical-feature effect from the target-update effect. `RESUME=1` restores the last saved episode boundary.

The default task is a 20 m path at 3 m/s with a 12 s deadline. F uses fixed CT terrain seed 1. J samples fresh courses each episode. Use `TERRAIN=hills`, `friction`, `terrain`, `slopes`, or `steps`. Evaluate new courses after F training with `CASE=F CASE_TEST=J COHORTS=layout`. Add `TERRAIN_TEST=steps` for frozen transfer to steps. Keep training settings unchanged when locating checkpoints.

Evaluation writes task JSON, speed/progress PNGs, and headless MP4s beside each checkpoint. Direct checkpoint evaluation is also available:

```bash
python eval_physical_fj.py --checkpoint-file /path/to/final.pt \
  --episodes 5 --clocks fixed,irregular --video-episodes 1 --device cpu --gl egl
```

For a short installation check, use the smoke commands in `PHYSICAL_FIRST.md`. Those settings perform 16 policy updates per method and supply no running-performance evidence. No costly training was run for this release.

The physical-first experiment has a new, explicitly instrumented sensor contract: exact q/v and causal generalized constraint-force readings. Acceleration and actuator-force labels are used during calibration. It learns three coefficients around the public robot XML; it does not infer an arbitrary unknown Hamiltonian.

The earlier F/J code, task, terrain, training, and checkpoint interfaces remain available. Their instructions are in [README_v3_2_1.md](README_v3_2_1.md). `scripts/run_path_speed.sh` still selects that earlier pipeline. This release also adds scale-aware residual recovery and diagnostics for that solver; `apply_finite_fix.py` can install only that correction into an existing folder while preserving its config and prepared data.

[VALIDATION.md](VALIDATION.md) records the executed checks. [CHANGELOG_PHYSICAL_FIRST.md](CHANGELOG_PHYSICAL_FIRST.md) lists the changes.
