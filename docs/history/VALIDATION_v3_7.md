# Cheetah MuJoCo baseline and transition audit

I compared five frozen controls using the supplied pure-CPHT checkpoints: learned-model feedback, MuJoCo least squares, contact-aware MuJoCo least squares, learned-model MPC, and MuJoCo MPC. There was no new policy training and no deployment parameter refit.

MuJoCo supplies physical dynamics and solves contact reactions. The controller must supply a task objective and choose motor commands. For a fixed current constraint port, the local reference solves

$$
\min_{a\in[-1,1]^6}\|W^{1/2}(M^{-1}Ba+b-d)\|^2+\lambda\|a\|^2,
\qquad b=\ddot q(a_0)-M^{-1}Ba_0.
$$

Here `d` is the existing controller's speed/clearance/pitch/joint acceleration target. Nine acceleration coordinates are controlled through six bounded motors. A small optimization residual therefore does not imply that the desired acceleration is reachable. The contact-aware variant finite-differences the full acceleration response and uses a local trust box.

## Full-course comparison

The task is 20 m at 3 m/s, with a 12 s deadline. Each cell below is one paired episode, evaluation seed 100000, starting at x = 0 m. Entries are signed COM displacement in meters; a negative value indicates backward motion. F uses its fixed course and J draws its course from the configured family. The two clocks share each case's physical layout.

The checked MPC budget was eight candidates, three elites, one CEM iteration, 20 predicted control intervals, and replanning every .1 s. This is smaller than the uploaded planner's 48 candidates/eight elites/three iterations/.04 s replanning. Thus these learned-MPC results do not reproduce the original video planning budget.

| Control | F fixed | F irregular | J fixed | J irregular |
|---|---:|---:|---:|---:|
| learned_feedback | -2.567 | -1.505 | 1.124 | 1.590 |
| mujoco_ls | -2.750 | -1.265 | 1.019 | 0.916 |
| mujoco_fd_ls | -0.940 | -0.011 | -0.484 | 0.057 |
| learned_mpc | 4.392 | 2.741 | 3.201 | 1.727 |
| mujoco_mpc | 7.198 | 4.578 | 13.255 | 6.423 |

All 20 runs reached the time limit without finishing the course. MuJoCo MPC made more progress than learned-model MPC in each of these four paired episodes. The simple local controls did not establish forward terrain running; in F they moved backward from the starting region. These results diagnose the selected motion representation, feedback, and planning budget. They do not establish terrain infeasibility or population-level performance.

The native controls use privileged MuJoCo mechanics/contact information. Native MPC also knows the full compiled current terrain/friction. The learned controller receives its causal state/force history and inferred environment. MPC variants also differ in feedback mechanics, contact closure, integration, and scan-coverage penalty. This is a practical reference comparison rather than a single-factor model ablation.

A converged local optimization can still produce poor locomotion. For fixed-clock F, the MuJoCo least-squares projected-gradient residual averaged approximately 1.9e-11, while the weighted acceleration-target residual averaged 43.8 in the declared coordinate scales. The contact-aware variant reduced the mean local linearization error from about 30.0 to 8.1 in those scales, and still failed to run forward.

## What the trained model captured locally

I exported four independent local terrain-state audits near x = 6 m: F/J with fixed/irregular holds. A causal .2 s feedback warmup precedes each sample; irregular warmup can end slightly later. These are new diagnostic replays, not recovered original video frames.

At these states, the fitted mass matrix and drift agree closely with the native model. The full learned contact/environment prediction has a larger error. The table reports numerical relative norms in this fixed coordinate chart; these norms combine translational and rotational coordinates and are not invariant physical norms.

| Sample | Hold (ms) | Mass relative error | Drift relative error | Constraint-port relative error |
|---|---:|---:|---:|---:|
| F fixed | 10.0 | 1.75e-07 | 9.01e-08 | 14.7% |
| J fixed | 10.0 | 9.40e-08 | 1.33e-07 | 18.3% |
| F irregular | 2.0 | 1.23e-07 | 6.91e-08 | 9.7% |
| J irregular | 2.0 | 9.70e-08 | 7.56e-08 | 6.5% |

The ordinary learned prediction uses estimated geometry/friction and the learned contact closure. Its error therefore combines environmental estimation, contact-model approximation, and integration effects. A second, explicitly privileged diagnostic supplies true current terrain samples and root-location friction while retaining the learned parameters. It does not consistently remove the error: for the fixed F sample, angular-velocity error changes from 1.331 to 1.280 rad/s; for fixed J it changes from .178 to .247 rad/s. Additional sensing alone does not resolve every local prediction error in these examples. One root-location friction value also cannot describe simultaneous contacts on different materials.

The uploaded runs all used `state` mode: 31 raw inputs and no external geometry sensor. The inferred terrain/friction state still updates from causal motion/force history. The 60-input augmented contracts require separately compatible trained networks. The original CT-SAC upload lacks `final.pt`, so I did not produce a new CT-SAC policy replay from its videos/logs.

The J log records one rejected model refit at decision 13824 because disjoint episode folds were unavailable. The stored model was retained and later refits succeeded; this is not evidence that the entire learning run failed.

## A concrete input/action/successor

`Cheetah_Transition_Input_Example.json` is the complete fixed-clock F example. The compact block below rounds displayed values; the JSON retains the recorded precision and includes the force port, uncertainty/validity fields, and every substep.

```json
{
  "time_seconds": 0.20000000000000004,
  "q": [
    6.040349,
    -0.211355,
    0.064353,
    0.705285,
    -0.290563,
    0.458457,
    0.012142,
    -0.550156,
    0.144148
  ],
  "v": [
    0.253653,
    -0.059122,
    -0.203886,
    1.224892,
    -1.269331,
    2.093858,
    0.274941,
    -0.930077,
    2.659691
  ],
  "constraint_port": [
    13.36161,
    275.263733,
    2.83018,
    44.591816,
    44.870827,
    8.894855,
    -83.622482,
    -47.728916,
    -17.038378
  ],
  "task_features": [
    0.699148,
    0.983333,
    0.3,
    1.0
  ],
  "estimated_friction": 0.6,
  "action_normalized": [
    1.0,
    -1.0,
    0.857028,
    1.0,
    -0.382003,
    0.981682
  ],
  "requested_hold_seconds": 0.01,
  "actual_hold_seconds": 0.01,
  "physics_substeps": 5
}
```

Raw feature order is q[0:9], v[9:18], generalized constraint force[18:27], remaining COM distance/path length[27], remaining time/horizon[28], target speed/10[29], next requested hold/reference time[30]. The JSON gives names and units for every entry. Root-z is a model coordinate with a fixed torso offset, rather than terrain clearance. The current constraint-force observation was computed under the previously held command. The newly selected command can change contact reaction.

For each substep, the audit checks the force equation and the installed backend's Euler update with implicit damping:

$$
M\ddot q+q_{\rm bias}=q_{\rm actuator}+q_{\rm passive}+q_{\rm applied}+q_{\rm constraint},
$$

$$
v_{k+1}=v_k+\delta(M+\delta D)^{-1}M\ddot q_k,\qquad q_{k+1}=q_k+\delta v_{k+1}.
$$

The fixed-hold examples compose five .002 s substeps. Both irregular examples execute one .002 s substep. Exact MuJoCo replay agrees with every observed q/v component, with zero reported maximum replay difference. Force-balance defects are at most 1.2e-11 and substep velocity defects at most 6.5e-14 in the recorded coordinate units.

The validator restores the complete integration state, checks the model hash/version, control/torque bounds, raw/successor vectors, timing, force balance, and every intermediate state. It rejects corrupted observations, timing, and actions. Its binary model snapshots require MuJoCo 3.17.0. A passing backend replay validates the data path; learned-model prediction remains a separate test.

## Reproduce without training

Unzip `TPHT_CT_RL_MuJoCo_Benchmark_v3_7_Reference.zip`, activate Torch, and install `requirements-physical-first.txt`. Keep the original extracted `sensor/` directory.

```bash
cd TPHT_CT_RL_Benchmark
export SENSOR_ROOT=/absolute/path/to/sensor
export MUJOCO_GL=egl

# Same small evaluation budget used in this report.
bash scripts/run_mujoco_reference.sh

# Export a fresh real terrain transition.
MODE=sample bash scripts/run_mujoco_reference.sh --clock fixed

# Validate a shipped sample; no training checkpoint needed for this command.
python -m mujoco_reference validate validation/reference_samples/fixed/F_transition.json
```

To evaluate with the uploaded planner's original search settings, add `--candidates 48 --elites 8 --iterations 3 --horizon-steps 20 --replan-seconds .04`. This costs more evaluation time and still performs no training. Use a new `OUT` directory. Use multiple independent training seeds before drawing a generalization claim. The current F/J tests vary regimes across episodes; no extra within-episode change was imposed.

The code bundle contains the full mathematical implementation note (`MUJOCO_REFERENCE.md`), all 20 results, action/state traces, plots, upload-log audit, four transitions, shared backend snapshots, and regression tests. Existing experiments remain available. Generated Python caches and extracted vendor caches are excluded. The original sensor archive and its trained checkpoints were not modified.

Validation covered 28 relevant tests, followed by seven final new-module tests including an added native-MPC isolation check (29 distinct tests in total). The final isolation check verifies that planning changes neither the live integration state, compiled model, observation, nor physical task time.

## Next diagnostic step

Keep mechanical calibration frozen while validating the contact/environment closure on these recorded transitions. Fit or compare the same held-step map used at deployment, separate geometry/friction belief errors from constitutive/contact errors, and scale translational and angular losses separately. In parallel, test a longer or richer motion planner against native MuJoCo. The available evidence identifies local contact/state-prediction and motion-selection gaps; it does not justify attributing every task failure to an unlearned mass matrix.

Sources: [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html), [MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html), and the checked dm-control 1.0.50 implementation. This extension uses native MuJoCo and finite differences; it does not claim a SuperMimic training reproduction.
