# MuJoCo benchmark v2: measured-data cases F and J

This revision adds the F/J experiments from Algorithms 3–4 and Section 7.7 of the measurements note.
It aligns their interval SAC targets with Sections 4.5–4.6 of the information-operator manuscript.
`cases/sources.json` records both documents' exact hashes and section references.

No policy training, observer fitting, or MuJoCo experiment was launched during preparation.
The new implementation covers the **CT and Op branches** of the notes.
It learns a predictive information operator and its numerical pH critic update.
A calibrated physical pH model, mechanical parameter identification, and physical safety certificates require the separate Phys branch.
The run metadata marks those claims as unresolved.

The original four-robot CT-RL suite remains available through `train.py`, `eval.py`, and `README_v1.md`.
The new F/J scripts implement the detailed **Cheetah** specification in Section 7.7.

## Quick start

Use the existing isolated environment and `requirements.txt` on the compute server.
These commands inspect the two cases without importing Torch or MuJoCo:

```bash
python train_fj.py --case F --experiment native --method ph_sac --dry-run
python train_fj.py --case J --experiment wind --method ph_sac --dry-run
bash scripts/run_F.sh
bash scripts/run_J.sh
```

The shell wrappers use `DRY_RUN=1` by default.
They default to seed 0, both subcases, and six comparison methods.
Use `EXPERIMENTS`, `METHODS`, and `SEEDS` to select a smaller job set.

Run the preparation stage once per experiment and training seed, then reuse it across methods:

```bash
# Case F: fixed plant/task, native measurements.
python prepare_fj.py --case F --experiment native --seed 0 --device cuda
python train_fj.py --case F --experiment native --seed 0 --method ct_sac --device cuda
python train_fj.py --case F --experiment native --seed 0 --method ph_sac --device cuda
python eval_fj.py --case F --experiment native --seed 0 --method ph_sac --device cpu

# Case J: hidden episode parameters and evolving wind.
python prepare_fj.py --case J --experiment wind --seed 0 --device cuda
python train_fj.py --case J --experiment wind --seed 0 --method interval_euler --device cuda
python train_fj.py --case J --experiment wind --seed 0 --method ph_sac --device cuda
python eval_fj.py --case J --experiment wind --seed 0 --method ph_sac \
  --cohorts iid,regime,task,wind,joint --device cpu
```

Equivalent shell launches for one complete stage are:

```bash
STAGE=prepare DRY_RUN=0 DEVICE=cuda bash scripts/run_F.sh
STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_F.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu bash scripts/run_F.sh

STAGE=prepare DRY_RUN=0 DEVICE=cuda bash scripts/run_J.sh
STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_J.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu COHORTS=iid,regime,task,joint bash scripts/run_J.sh
```

Add `EXPERIMENTS=wind` to the J wrapper when using the `wind` holdout cohort.
Preparation is an explicit stage. Training fails if its common artifact is missing.

## The four experiments

- **F/native:** fixed XML, gears, ground, load, speed target, and measurement law. Input is the native 17-component position/velocity observation.
- **F/position:** the same plant and task. Input contains eight positions with independent Gaussian noise of standard deviation 0.01 rad. A shared causal observer is fitted before policy comparison.
- **J/static:** friction and payload are drawn once per episode and remain hidden. The known speed descriptor conditions the actor, critics, and operator.
- **J/wind:** the static variation remains. An OU wind state evolves within the episode and loads an off-centre material point through its current Jacobian.

All cases retain nine generalized coordinates, nine velocities, and six commands in `[-1,1]`.
The backend checks motor gears `(120,90,60,90,60,30)` and records the installed XML hash.
The default physics step is 0.002 s. Decision holds range from 0.002 to 0.030 s.
The episode horizon is 10 s, with the source's 2,000-decision cap.
The clock can also use a fixed 0.01 s hold.
Its quantized upcoming hold is supplied to every method before action selection.
The policy controls the motor command; sensing and duration selection follow the declared protocol.

For J, the provided training grid is:

- Known speed targets: `6, 10, 14` m/s.
- Hidden sliding-friction multipliers: `0.6, 1.0, 1.4`.
- Hidden payloads: `0, 1, 2` kg.

These numerical grids are explicit benchmark choices. The notes define the parameter family without prescribing these grid values.
Validation and test grids have separate, disjoint values in `fjbench/config.py`.
The reward is `clip(torso_subtree_speed / known_speed_target, 0, 1)`.
Sliding friction is scaled on both sides of contact pairs.
The payload shares the torso centre of mass and preserves its radius of gyration by scaling inertia with mass.
Neither hidden parameter enters an actor input, replay feature, or task label.

For J/wind, the default is

$$
dw=-\kappa_w w\,dt+\sigma_w\,dW_t,\qquad
F=[3w,0,0]^\top,\qquad \tau_{\rm ext}=J_c(q)^\top F.
$$

The code uses the exact OU marginal update and holds its force during each physics substep.
It recomputes the off-centre point and Jacobian each substep.
The Brownian increment acts on wind. The mechanical momentum receives the time-integrated force.
This physical discretization still needs server refinement tests for an accuracy claim.

## Information, measurements, and training stages

The raw replay record keeps current/successor observations, past and current commands, received endpoint reward, interval return, actual duration, queued next duration, known task, clock, and separate termination types.
It contains no hidden friction, payload, wind, or privileged velocity label in the position-only experiment.
Hidden truth is written only to evaluation audit fields.

F/native uses the normalized current measured state.
F/position and both J cases use a causal GRU over 16 raw observation/action/reward/duration frames.
Every method shares the same fitted information map within a case and seed.
The GRU is a predictive memory approximation. Its output is not an exact Bayesian posterior.
The reported closure score compares short- and long-history predictions; it is not the posterior KL certificate in equation (72).

Preparation collects 30,000 random-action decisions by default.
It separates training and validation by episode before fitting.
The observer fits a joint next-observation/reward predictive loss and a predictive history-consistency term.
After that fit, the observer is frozen and the operator is fitted on the training partition.
The validation episodes remain excluded from all later policy and operator gradients.

The operator predicts reward channels and positive normalized transition rows over a fixed information-node bank.
Known task strata have separate transition support.
Its loss includes reward error, soft transition-label loss, and held-out terminal-value probe backups.
Probes reuse the same physical records. They create no additional independent measurements.
F uses one nominal model member; J uses three independently fitted members.
Their conditional probabilities are averaged with a common current action.
The ensemble is a predictive approximation and supplies no calibrated bath temperature or exact parameter posterior.

The default comparison freezes the common observer and operator across methods.
This isolates critic-update effects on the same predictive model.
For an additional online-operator experiment, pass `--config cases/online_operator.json` to every compared method.
Its candidate model updates are checked on separate calibration episodes and can be rejected.
Rejected updates still count toward computation cost.
The observer and node bank stay fixed; all history embeddings are computed from raw replay.
Full online observer replacement and a Phys-branch model need an additional implementation and validation contract.

## Methods and the frozen critic block

The methods are `ct_sac`, `ct_sac_i`, `ct_td3`, `interval_sac`, `model_sac`, `interval_euler`, `ph_sac`, and `ph_direct`.

- `ct_sac` and `ct_td3` use the source target equations with the common F/J information interface.
- `ct_sac_i` replaces the source reward term by the measured interval return divided by the reference-duration ratio.
- `interval_sac` uses measured interval transitions and neural continuation values.
- `model_sac` uses the common learned transition rows with neural continuation values.
- `interval_euler`, `ph_sac`, and `ph_direct` use exactly the same frozen finite twin-critic operator.

`ph_sac` is labelled CPHT-SAC for one nominal member and TPHT-SAC for the ensemble.
The source-equation adapters use fixed entropy and the common observer.
They therefore have a separate label from the unchanged published runners retained in `README_v1.md`.
CT-TD3 retains delayed actor/target updates, collection exploration, and separate target smoothing.
Its objective has no SAC entropy term.

One update follows this algorithm:

1. Read a measured transition and retain its known task and actual hold.
2. Recompute causal information from raw windows with the shared frozen observer.
3. For an online Op run, fit a candidate predictive model and test it on held-out calibration episodes.
4. Freeze the admitted model, observer, actor, entropy convention, nodes, and action quadrature.
5. Apply the selected source target, interval backup, relaxed Euler solve, or pH residual solve.
6. Fit both critics. Record the fitting error and storage jump. In strict mode, apply the complete numerical gate before adopting the fitted update.
7. Update the actor and targets on their declared schedule. A new actor, model, or quadrature snapshot starts a new residual chart with zero solver momentum.

The finite pH block evolves both free critic heads.
Candidate critic values change inside the operator throughout recovery.
The exact frozen-policy inverse reduces to one information-node linear solve; it retains the distinct target residuals of both heads.
`DESIGN_FJ.md` gives the equations, data boundaries, and equation-to-module mapping.

The phase budget defaults to three midpoint steps per block.
Logs report residual, momentum, convergence status, inverse error, recovery storage port, and post-fit storage debt.
Budget exhaustion remains visible. It is not reported as convergence.
Set `require_solver_tolerance=true` and a larger `phase_steps` budget for a stricter finite-solve study.
That choice increases the inner cost and can reject updates.

The default `fit_admission=empirical` retains neural regression progress and records any failed fitting or returned-storage gate.
It supplies an experimental RL run, with explicit numerical debts.
Strict mode restores both critic and optimizer when the complete gate fails and skips that actor update.
Use `--config cases/strict_solver.json` for that separate numerical-admission study.
A small regression budget can produce many strict rejections; report them with the control results.
The finite phase storage identity and the returned neural storage check have separate log fields.

## Entropy and reward units

The reference time defaults to 0.01 s. The raw source reward is the normalized task reward.
The interval return is its discounted integral in reference-time units, with the same physics quadrature across methods.
For `entropy_convention=start`,

$$
\bar h=h/t_{\rm ref},\qquad \alpha_{CT}=\alpha_{\rm reference},\qquad
\alpha_h=\bar h\alpha_{CT}.
$$

This implements the finite-interval convention in manuscript equation (37).
The optional `discounted` convention instead uses
`alpha_reference * (1 - gamma_reference**(h/reference_time)) / (-log(gamma_reference))`.
No method uses an average entropy coefficient over unequal holds.
The current hold determines the actor coefficient; the successor hold determines successor policy entropy.
All action log densities use the same squashed-Gaussian convention.

Continuing tasks bootstrap through time-limit truncation.
`boundary=finite_horizon` includes the clock in the information map and removes terminal continuation.
Set `episode_step_cap` to at least `ceil(horizon/min_hold)` in that variant so a second, unobserved decision-count deadline does not bind.
Physical termination always removes future interval reward and entropy.
Source-code terminal masking and the small duration floor remain separately logged.
Parameter averaging also records its deviation from exact prediction averaging.

## Plans, budgets, recovery, and Slurm

The supplied `plans/fj_prepare.jsonl` has 48 shared preparation jobs.
`plans/fj_train.jsonl` has 384 method/case/seed jobs.
They cover four experiments, eight methods, and twelve independent training seeds.
Plan creation launches no simulation.

```bash
python make_fj_plan.py --kind prepare --out plans/fj_prepare.jsonl
python make_fj_plan.py --kind train --out plans/fj_train.jsonl
python prepare_fj.py --plan plans/fj_prepare.jsonl --index 0 --device cuda
python train_fj.py --plan plans/fj_train.jsonl --index 0 --device cuda
```

The 1,000,000-decision budget includes all 30,000 calibration decisions, including calibration validation.
Each method receives the same eligible calibration training records.
Preparation time is split into acquisition, observer fitting, and operator fitting.
Operator fitting is charged to methods that use that model; source methods report their common observer cost separately.
Training records physical steps, observations, reward samples, online model work, inverse solves, critic gradients, actor gradients, rejected fits, and update time.
These are matched interaction budgets. Inner numerical and neural fitting costs can differ.

Resume works at an episode boundary:

```bash
python train_fj.py --plan plans/fj_train.jsonl --index 0 --device cuda --pause-hours 10
python train_fj.py --plan plans/fj_train.jsonl --index 0 --device cuda --resume
```

The checkpoint includes replay, model/critic/actor optimizers, generators, counters, the validation sampler, and the next episode seed.
Use the same stack and hardware class for numerical reproducibility.
The final policy checkpoint contains its public observer/model components and requires no absolute calibration path for evaluation.

Use `scripts/slurm_fj.sh` with your TACC account, partition, GPU request, and array range.
Set `STAGE=prepare` for indices `0-47`, then run the training array `0-383` after preparation completes.
Set `STAGE=eval DEVICE=cpu` for frozen-policy evaluation.
The script has no fixed cluster account or partition.

## Evaluation and claims

Use paired reset seeds `50000,...` for validation and `100000,...` for final evaluation.
Evaluation uses the squashed mean action, fresh causal history, and zero learning updates.
J cohorts separate in-distribution evaluation, held-out physical regimes, held-out known tasks, unseen wind laws, and their joint change.
Each result keeps the task descriptor. Average within tasks before averaging the declared task family.
Then aggregate independent training seeds:

```bash
python aggregate_fj.py --plan plans/fj_train.jsonl --root runs \
  --cohort iid --clock irregular --reference interval_euler --out results/fj_iid.json
```

Evaluate all jobs in the requested plan first.
The aggregator rejects missing cases, mixed configurations, unequal paired seeds, and evaluations whose checkpoint hash is stale.
It reports paired differences and bootstrap intervals over training seeds.

The finite solver tests establish algebraic properties of a declared positive operator.
They do not establish sufficient learned memory, a uniform model-error bound, global neural convergence, physical passivity, or optimal policy return.
Held-out task performance is measured through the conditioned actor.
The training node bank has no new-task lifting certificate.
Those unresolved claims are explicit in every run manifest.

See `VALIDATION.md` for preparation checks. The optional server checks are:

```bash
python -m unittest discover -s tests -v
python check_fj_backend.py
```

The backend check takes two zero-action decisions per experiment and performs no training.
