# Running the experiment

One launcher runs the whole experiment at three scales:

```bash
./run_experiments.sh smoke            # all stages, about a minute on a CPU
./run_experiments.sh pilot            # all stages, a small real comparison (hours; see Compute cost)
./run_experiments.sh full             # prints the plan only (156 training runs)
./run_experiments.sh full all         # runs the paper-scale protocol
```

The second argument picks a stage: `plan`, `prepare`, `gradcheck`, `train`,
`eval`, `report`, `verify` or `all`. Each scale reads `experiments/<scale>.json`
and writes to `artifacts/<scale>/`, `runs/<scale>/` and `reports/<scale>.json`
(plus a Markdown table next to it). Finished stages are reused when you
rerun. `RESUME=1` continues a partly trained run from its last episode
boundary.

## What one experiment does

```text
sense ──► measure ──► calibrate pH model ──► plan/learn through it ──► bounded feedback ──► act
  ▲                                                                                         │
  └──────────────────────────────── MuJoCo cheetah, 2 ms substeps ◄─────────────────────────┘
```

1. **Sensing** (`environment_belief/`, `physical_fj/mechanics.py`). The robot is
   measured through exact joint positions, velocities and constraint-force ports
   (31 channels). The information mode then decides what else a policy sees:

   | Mode | Policy input |
   | --- | --- |
   | `state` | the 31 channels; terrain is inferred from contact history only |
   | `state_belief` | 60 channels: plus a causal terrain/friction belief built from the same history |
   | `sensed_belief` | 60 channels: the belief also assimilates noisy, delayed height/slope rays (period, delay and dropout set in `learning.sensing`) |

   No policy sees the terrain map, true friction, future readings or the next
   hold length.

2. **Measurement and calibration** (`physical_fj/prepare.py`,
   `pure_cpht/learning.py`). For each sensing mode, case and seed, one shared run
   of random commands records instrumented accelerations and force ports. From
   these, the code fits the port-Hamiltonian model
   H = ½ pᵀM(q)⁻¹p + U(q) with 32 positive mechanical coefficients and a
   4-parameter contact closure. All methods share this calibration and its
   budget.

3. **The pH model as the learning engine** (`cpht_compare/rollout.py`,
   `cpht_compare/learning.py`). Every pH method imagines held-action rollouts
   through the frozen calibrated model and optimizes the task return there:

   | Method | What is learned or searched through the pH rollout |
   | --- | --- |
   | `cem` | cross-entropy search over 19 periodic-gait coefficients, each decision |
   | `shooting` | gradient ascent on the same coefficients (backpropagation through the rollout) |
   | `selector` | a neural policy that outputs the coefficients, trained by backpropagation through time |
   | `selector_value` | the selector plus a learned task value at the rollout horizon, fitted by TD(λ) |

4. **Feedback control** (`pure_cpht/controller.py`, `pure_cpht/feedback.py`).
   The selected gait becomes six motor commands through an acceleration-feedback
   QP under the calibrated mass matrix, solved exactly in the [-1, 1]⁶ box (no
   clipping).

5. **Baseline** (`cpht_compare/baseline.py`, `physical_fj/train.py`). `ct_sac`
   is model-free CT-SAC with the source rate target (paper eq. 2.26). It uses the
   same sensors, the same calibration transitions (as initial replay) and the
   same total decision budget. It is trained once per mode, case and seed, and is
   paired with the pH methods at the first horizon.

6. **Evaluation and report** (`cpht_compare/pipeline.py`, `cpht_compare/report.py`).
   Every run is evaluated on paired episodes. The report checks that all
   methods in a group share courses, calibration, sensing and task, and
   summarizes per training seed.

## Task and conditions

The task is a finite path on the pinned CT-RL terrain heightfields: cover 20 m
at a 3 m/s target within 12 s, with hard fall and goal events (torso clearance
below 0.20 m or pitch above 1.5 rad ends the episode).

- **F** keeps one fixed course (course seed 1).
- **J** draws a fresh course every episode.

The robot itself is fixed in both cases. Evaluation crosses three settings:
test case (`same`, or `J` for F-to-J transfer), clock (`fixed` 10 ms holds or
`irregular` holds from the two-tail law, paper eq. 4.31) and condition:

| Condition | What changes |
| --- | --- |
| `nominal` | the training distribution |
| `layout` | held-out course layouts |
| `friction_change` | hidden friction switches mid-episode |
| `sensor_shift` | more sensor delay and dropout (only in `sensed_belief`) |
| `steps` | step terrain |

**Holds are unannounced.** Policies and imagined rollouts see the nominal 0.01 s
hold, and the actual hold is drawn after the command is chosen (paper §2.6).
`DISCLOSE_HOLDS=1` restores the v3.7 interface as an ablation; it changes the
run identities, so both variants can coexist.

## Scales

| | smoke | pilot | full |
| --- | --- | --- | --- |
| Cases | F, J | F, J | F, J |
| Seeds | 0 | 0 | 0, 1, 2 |
| Sensing modes | state_belief | sensed_belief | state_belief, sensed_belief |
| Planning horizons | 0.02 s | 0.2 s | 0.2, 0.4, 0.6 s |
| Decisions per run (incl. calibration) | 68 (64) | 6,096 (1,024) | 100,000 (4,096) |
| Training runs | 10 | 10 | 156 |
| Evaluation | 1 episode, ≤ 8 decisions; same case | 2 episodes, full course; same case | 5 episodes, full course; same case and F→J |
| Purpose | checks that every stage runs | first real comparison | paper protocol |

Smoke results rank nothing: every episode is cut off after 8 decisions.

### Compute cost

Measured on one CPU thread at pilot settings (0.2 s planning horizon). Each run
is single-threaded, so independent runs can execute in parallel.

| Step | Cost |
| --- | --- |
| Shared calibration, 1,024 decisions | 15 s |
| `ct_sac` decision | 0.03 s |
| `selector`, `selector_value` decision | 0.12 s |
| `cem` decision | 0.25 s |
| `shooting` decision | 2.0 s (8 gradient steps through the rollout) |

A pilot training run spends 5,072 policy decisions: about 2.5 min for
`ct_sac`, 10 min for the selectors, 21 min for `cem` and 2.8 h for
`shooting`. Evaluation uses the same per-decision cost. A full-course episode
is up to 1,200 decisions on the fixed clock and about twice that on the
irregular clock. Costs grow with the planning horizon.

The full scale (100,000 decisions per run) is meant for a cluster: run
`prepare` once, then split `train` and `eval` into jobs with the selection
variables below. Every job reuses the shared calibration artifacts, and
`DEVICE=cuda` moves the rollouts to a GPU.

## Selecting part of a scale

Environment variables override the profile without editing it:

```bash
METHODS=selector,ct_sac CASES=J SEEDS=1 ./run_experiments.sh full train
INFORMATION_MODES=sensed_belief CONDITIONS=nominal,steps ./run_experiments.sh pilot eval
DEVICE=cuda KEEP_GOING=1 ./run_experiments.sh full train
```

Also available: `HORIZONS`, `CLOCKS`, `TEST_CASES`, `EPISODES`,
`EVAL_MAX_DECISIONS`, `OVERWRITE=1` (re-evaluate after a code change) and
`PROFILE=path.json` (your own scale). A SLURM array can run one
`METHODS`/`SEEDS` combination per task after a single `prepare` job.

## Checks that need no training

```bash
./run_experiments.sh smoke verify
SENSOR_ROOT=/path/to/sensor ./run_experiments.sh smoke verify
```

`verify` replays the four recorded cheetah transitions in
`validation/reference_samples/` bit-for-bit and checks their force balance.
It then runs `tools/paper_checks.py`, which compares the manuscript's
equations with the code and MuJoCo. With `SENSOR_ROOT`, it also runs the
privileged MuJoCo reference controllers on your existing trained checkpoints
(see [MUJOCO_REFERENCE.md](MUJOCO_REFERENCE.md)). `gradcheck` compares the
analytic pH-rollout gradient with finite differences.

## Outputs

```text
artifacts/<scale>/<mode>/<case>/seed<s>_<id>/    calibration data, fitted pH model
runs/<scale>/<mode>/<case>/<h|model_free>/<method>/seed<s>_<id>/
    run.json, train.jsonl, episodes.jsonl, latest.pt, final.pt
    evaluations/<case>to<test>_<clock>_<condition>_<sig>.json
reports/<scale>.json, reports/<scale>.md         paired, seed-level summary
```

Each evaluation records task progress, speed error, falls, prediction and
solver residuals, motor saturation, the belief trace and work counters. The
report keeps task failures separate from evaluation-budget censoring and
numerical flags.
