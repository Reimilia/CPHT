# Cheetah CPHT comparison, v3.7

This revision adds a no-training MuJoCo reference comparison for the supplied trained `sensor.zip` runs. It includes bounded current-state least squares, contact-aware finite-difference feedback, native MuJoCo MPC, and saved learned-model feedback/MPC. It also exports real input/action/transition examples with force balance, time stepping, and exact backend replay checks.

See [MUJOCO_REFERENCE.md](MUJOCO_REFERENCE.md) for equations, information access, and commands. Checked F/J results are under `validation/mujoco_reference/`; portable input/transition examples are under `validation/reference_samples/`.

```bash
SENSOR_ROOT=/absolute/path/to/sensor \
  bash scripts/run_mujoco_reference.sh

SENSOR_ROOT=/absolute/path/to/sensor MODE=sample \
  bash scripts/run_mujoco_reference.sh
```

The comparison runs frozen controllers and model-based control. It performs no training. Earlier experiments below remain available.

## Differentiable training experiment from v3.6

This revision adds a SuperMimic-inspired differentiable training experiment to the v3.5 sensing benchmark.

| Method | Motion decision | Learned task component |
| --- | --- | --- |
| `cem` | Cross-entropy search over 19 periodic-reference coefficients | Search state |
| `shooting` | Gradient shooting over the same coefficients | Search state |
| `selector` | Neural selector predicts those coefficients | Selector weights |
| `selector_value` | Same selector with terminal task-value continuation | Selector and value weights |

All four use the same calibrated mechanical/contact model, bounded six-input feedback, predicted task objective, sensing contract, and current-hold disclosure. Model parameters stay frozen after shared calibration. The existing CT-SAC, CPHT-SAC, and original `pure_cpht` runners remain available.

The new `cem` branch uses the common task objective for this ablation. The original v3.5 `pure_cpht` runner retains its earlier planning cost.

## Quick start

Activate Torch and install the simulator dependencies:

```bash
python -m pip install -r requirements-physical-first.txt
```

Inspect the plan. The launcher is dry by default:

```bash
STAGE=all DRY_RUN=1 bash scripts/run_cpht_comparison.sh
```

Run the inexpensive installation check:

```bash
PROFILE=cases/cpht_comparison_smoke.json \
RUN_ROOT=runs/cpht_smoke ARTIFACTS=artifacts/cpht_smoke \
REPORT=reports/cpht_smoke.json \
STAGE=all DRY_RUN=0 DEVICE=cpu bash scripts/run_cpht_comparison.sh
```

It uses 64 shared calibration decisions and four subsequent policy decisions per method/case. Evaluation stops after eight decisions; completion remains unassessed.

For the intended server experiment:

```bash
export CASES=F,J
export SEEDS=0,1,2
export HORIZONS=0.2,0.4,0.6
export INFORMATION_MODES=state_belief,sensed_belief
export DEVICE=cuda

STAGE=prepare DRY_RUN=0 bash scripts/run_cpht_comparison.sh
STAGE=gradcheck DRY_RUN=0 bash scripts/run_cpht_comparison.sh
STAGE=train DRY_RUN=0 bash scripts/run_cpht_comparison.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu bash scripts/run_cpht_comparison.sh
STAGE=report DRY_RUN=0 bash scripts/run_cpht_comparison.sh
```

Keep the same profile and sweep variables across stages. `METHODS=cem,shooting` selects the search comparison. `RESUME=1` resumes at a saved episode boundary. `TEST_CASES=same,J` includes F-to-J transfer. `CONDITIONS` and `CLOCKS` select tests. `OVERWRITE=1` refreshes an evaluation after an intentional code/checkpoint change.

The full profile declares 100,000 total decisions including 4,096 calibration decisions. Its complete sweep is substantial; it runs only with `DRY_RUN=0`.

## Outputs and limits

`reports/cpht_comparison.json` and its Markdown companion contain paired, training-seed-level summaries, failure diagnostics, censoring, and missing-run information. Each evaluation retains task progress, speed error, falls, model/solver residuals, action saturation, causal assimilation traces, and work/timing counters.

See [VALIDATION_v3_6.md](history/VALIDATION_v3_6.md) for the local checks and [DIFFERENTIABLE_CPHT.md](DIFFERENTIABLE_CPHT.md) for equations and assumptions. Short smoke checks do not rank trained policies or establish full-course generalization.

The code consulted is the independent [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) repository at commit `ba6105349069bf040f1fdad60873d13138d505a2`. Its README identifies it as unofficial. The adaptation uses frozen-model BPTT and terminal-value/TD(lambda) patterns with our structured predictor; it does not port G1 checkpoints. Attribution and the upstream MIT license are in `third_party/supermimic_reproduced/`.

Old generated plans, media, plots, and diagnostic logs were removed from the distributable. Legacy runnable code, configurations, vendor source, and the small feedback-model fixture remain. The previous guide is [README_LEGACY.md](history/README_LEGACY.md).

