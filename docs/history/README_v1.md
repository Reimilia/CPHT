# CT-RL and TPHT: a MuJoCo benchmark protocol

This bundle provides train and evaluation scripts for the four robotics tasks in the CT-RL control suite.
Long training jobs have **not been run** for this delivery.
The bundle includes the supplied CT-RL source archive and records its SHA-256 hash.

Use two experiment tracks. Keep their labels in every result.

1. **Released track.** Run the eight released algorithms with their published CSV settings and original training entry points.
2. **Matched track.** Use one neural actor–critic implementation to compare time-aware SAC, a CT-SAC variant, and finite Picard/TPHT updates.

The matched TPHT method is a **new finite-collocation extension** of the initial Acrobot project.
Its finite solver has explicit numerical checks.
Its neural approximation and control performance require experiments on the server.

## 1. What the benchmark compares

Use the exact DMC task identifiers:

- `cheetah-run`: 1,000,000 training decisions; 10 s episode horizon; 0.002 s physics step.
- `walker-run`: 1,000,000 decisions; 25 s horizon; 0.0025 s physics step.
- `humanoid-walk`: 2,000,000 decisions; 25 s horizon; 0.005 s physics step.
- `quadruped-run`: 1,000,000 decisions; 20 s horizon; 0.005 s physics step.

These environments use DeepMind Control with the MuJoCo backend.
Their observations and reward functions define the task.
Keep these task definitions when comparing with the CT-RL paper.
Gymnasium locomotion tasks have separate definitions and scores.

The released track includes `ct_sac`, `ct_td3`, `q_learning`, `cppo`, `sac`, `td3`, `ppo`, and `trpo`.
It preserves each method's actor, critic, entropy update, replay settings, and CSV hyperparameters.
The source contains financial modules because its native runners import them.
The benchmark accepts only the four control tasks and runs no financial experiment or data download.

One callback adds a CT checkpoint after the final learning update.
The original periodic CT checkpoint at the budget is saved during collection, before that update.
This difference is one update for off-policy methods and one rollout update for CPPO.
Use `--checkpoint final` for the post-training comparison.
Use `--checkpoint native_final` to inspect the original periodic checkpoint's effect on released results.
The source files and numerical training updates remain unchanged.

The matched track supplies these executable adapters:

- `sac_smdp`: soft actor–critic with observed-duration discount and an integrated reward.
- `ct_sac_corrected`: the released CT-SAC first-order target form, with a terminal correction and shared training settings.
- `ct_picard`: relaxed Picard iteration on the frozen finite operator, with relaxation 0.5.
- `tpht`: a damped Hamiltonian residual step, followed by inverse recovery and neural fitting.
- `collocation_direct`: direct solution of the same finite policy-evaluation operator.

Optional ablations are `picard`, with relaxation 1, and `tpht_damping`, with zero solver interconnection.
The default matched plan includes the first five methods.
The direct-solve baseline exposes the cost and effect of inverse recovery.

## 2. Rules for a fair comparison

Use 12 independent training seeds, `0,...,11`, for the final results.
Pair these seeds across methods. Each method will collect its own policy-dependent trajectories.
Matched methods share episode reset seeds, clock distributions, warmup actions, and collection-noise streams.
They receive only the same DMC observation and action spaces.

For the matched track, keep the actor and both critics at widths `(400,300)`.
Use batch size 256, replay capacity 300,000, and 10,000 warmup decisions.
Apply one actor update and one critic update per collected decision.
Keep the entropy rate at 0.2 and target-network averaging at 0.005.
Use a native-step discount of 0.98; use 0.99 for Humanoid.

The released track retains its original settings, including method-specific differences.
For example, the released Humanoid settings use a batch size of 64 for several methods.
Treat that track as a reproduction reference. Use the matched track to study update design.

Count all training decisions, including warmup, toward the budget.
Record physical substeps and simulated seconds as additional data costs.
Record update time, total training time, network forward batches, kernel distance pairs, operator products, and linear solves.
The matched GPU timer synchronizes before and after each learning update.
Training time in the native runners also includes their inline validation and file output.

Equal decision budgets specify the primary comparison.
The additional cost records support a separate compute comparison.
Use the same hardware and software stack for that comparison.
The boundary-pause option supports job recovery; its episode-length overshoot requires an explicit record when used as a time budget.

Use the **final checkpoint at the specified training budget** for the primary score.
Use validation seeds `50000,...` for hyperparameter selection.
Use test seeds `100000,...` once the settings are fixed.
Average 20 paired test episodes per trained policy and test case.
Report a deterministic policy first; label stochastic-policy results separately.

The native runner can save a validation-best model. The evaluation script selects the final-budget checkpoint.
It creates a fresh DMC instance for each evaluation episode.
This step also seeds the task's physical reset, which the released wrapper's `reset(seed=...)` does not change.

## 3. Initialize the server project

Use Python 3.11 or 3.12. Create and activate an isolated environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin the simulator, DMC, Torch, Gymnasium, and baseline package versions.
Select a Torch wheel that supports the server's CUDA driver when your cluster needs a custom wheel index.
Every run records the installed versions. The original paper did not supply this exact package lock.
Thus the source is fixed, while exact paper-score reproduction remains a numerical experiment.

These quick checks collect no training rollout:

```bash
python -m unittest discover -s tests -v
python check_backend.py
```

The backend check takes two zero-action decisions per robot with a point-force environment.
It prints the robot dimensions, applied-force body, and directly forced degrees of freedom.

## 4. Create and inspect a plan

The bundle contains `plans/released.jsonl` and `plans/matched.jsonl`.
They contain 384 released runs and 240 matched runs, respectively.
Each line defines one run. The index starts at zero.
Creating or inspecting a plan starts no training.

```bash
python make_plan.py --track released --out plans/released.jsonl
python make_plan.py --track matched --out plans/matched.jsonl
python train.py --plan plans/matched.jsonl --index 0 --device cpu --dry-run
python eval.py --plan plans/matched.jsonl --index 0 --dry-run
```

For a smaller initial server comparison, create one Cheetah plan:

```bash
python make_plan.py --track matched --tasks cheetah-run \
  --algos sac_smdp,ct_sac_corrected,ct_picard,tpht,collocation_direct \
  --seeds 0 --out plans/cheetah_initial.jsonl
```

The algorithm and seed filters apply to either track.
Use `--rank second` or `--rank third` to inspect the released alternatives.
These names retain their published ranking and selection history.

## 5. Train one remote job

```bash
python train.py --plan plans/matched.jsonl --index 0 --root runs --device cuda
python train.py --plan plans/released.jsonl --index 0 --root runs --device cuda
```

The scripts write a run manifest, logs, status, and checkpoints.
Matched runs save small policy snapshots at each 100,000-decision boundary.
They save the full replay and optimizer state to `latest.pt` at an episode boundary after that interval.
The final policy is saved at the exact decision budget.

For a matched job, request a pause before the scheduler time limit:

```bash
python train.py --plan plans/matched.jsonl --index 0 --root runs \
  --device cuda --walltime-hours 10
python train.py --plan plans/matched.jsonl --index 0 --root runs \
  --device cuda --resume
```

The pause completes the current episode before it saves state.
Resume restores replay, optimizers, random generators, counters, and the next episode seed.
It removes log entries and policy snapshots beyond the recovery checkpoint.
Use the same software stack and hardware class when testing numerical reproducibility.
Check for an active process before removing a stale `run.lock` after a hard scheduler kill.

The released models save network weights. Their original runners provide no exact replay-and-optimizer resume path.
Give those runs enough scheduler time to finish.
The wrapper checks that their final-budget checkpoint exists and marks failures explicitly.

Use the supplied Slurm templates with your site's resource arguments:

```bash
export BENCH_ROOT="$PWD"
export PLAN="$PWD/plans/matched.jsonl"
export RUN_ROOT="$PWD/runs"
sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  --gres=gpu:1 --cpus-per-task=4 --time=24:00:00 \
  --array=0-239%8 slurm_train.sh
```

Adjust the GPU request to your TACC system's Slurm syntax.
The scripts contain no fixed account or partition.
Set `RESUME_MATCHED=1` only for an array subset with existing `latest.pt` files.

## 6. Evaluate the frozen policies

For direct comparison with the paper's task wrapper and reward sum:

```bash
python eval.py --plan plans/released.jsonl --index 0 --root runs \
  --protocol paper --episodes 20 --device cpu
```

For one common evaluation environment across released and matched policies:

```bash
python eval.py --plan plans/released.jsonl --index 0 --root runs --protocol common
python eval.py --plan plans/matched.jsonl --index 0 --root runs --protocol common
```

The default evaluates both irregular and regular decision clocks in the nominal environment.
Each evaluation creates fresh simulator instances and makes zero learning updates.
Use the same protocol, clocks, contexts, episode seeds, and policy mode for every compared method.

To test environmental transfer, keep the learned parameters fixed:

```bash
python eval.py --plan plans/matched.jsonl --index 0 --root runs \
  --protocol common --clocks irregular,regular \
  --contexts nominal,damping_1p5,mass_1p2,point_drag,point_pulse
```

The common environment defines these cases:

- `damping_1p5` multiplies joint damping by 1.5.
- `mass_1p2` multiplies body mass and body inertia by 1.2 and recomputes MuJoCo constants.
- `point_drag` applies a relative-flow force to one distal body.
- `point_pulse` applies a horizontal pulse from 25% to 40% of the specified episode horizon.

The force body is selected deterministically from the deepest bodies with positive mass.
The application point is 5 cm from its centre of mass in local coordinates.
At each physics substep, the code computes the world point and its Jacobian:

\[
 x_c(q)=x_b(q)+R_b(q)r_c,\qquad
 v_c=J_c(q)v,\qquad \tau_{\rm ext}=J_c(q)^\top F_c.
\]

For relative flow, \(F_c=0.05m_{\rm total}([1,0,0]^\top-v_c)\), in SI units.
The pulse magnitude is 5% of total robot weight.
The force enters MuJoCo through `qfrc_applied`; it is recomputed at each physics substep.
The recorded `directly_forced_dofs` counts nonzero entries of this generalized force.
Inertial coupling can change motion at additional joints.

The standard DMC task still supplies the contact dynamics and robot actuation.
The common environment accumulates reward at physics substeps and separates termination from a time-limit truncation.
Its dynamics calls and reward quadrature define an additional evaluation protocol.
Keep its scores separate from scores obtained through the original paper wrapper.

These cases test transfer across timing, inertial parameters, and external forces.
Each robot has its own trained policy. Transfer between robots with different state and action dimensions needs an additional policy architecture.
This benchmark supplies no test-time calibration or learned mechanical pH model.

## 7. Define and aggregate the scores

The evaluator records two scores:

\[
 R_{\rm decision}=\sum_i r_i^{\rm endpoint},\qquad
 S_{\rm time}=\frac{1000}{T_{\max}}\sum_j r_j\,\Delta t_j.
\]

Use `raw_decision_return` with the `paper` protocol for the released comparison.
Use `time_score` with the `common` protocol for timing and environmental transfer.
The common score integrates rewards at physics substeps.
The paper protocol also records an endpoint quadrature estimate of the time score.
The fixed denominator penalizes early episode endings; the file also records actual horizon coverage.

Average episodes within each trained policy first.
Then compute the mean, interquartile mean, and 95% bootstrap interval across independent training seeds.
Pair seed-level means when comparing two methods.
The aggregation script treats an absent evaluation as missing data and stops by default.
It also rejects duplicate configurations for a task, method, and training seed.

```bash
python aggregate.py --plans plans/matched.jsonl --root runs \
  --protocol common --metric time_score --reference ct_sac_corrected \
  --out results/matched_final.json
python aggregate.py --plans plans/released.jsonl --root runs \
  --protocol paper --metric raw_decision_return --reference ct_sac \
  --out results/released_final.json
```

Use both plans in a common-protocol aggregation after evaluating both through that protocol.
Use `--checkpoint 200000` in evaluation and aggregation to inspect a common saved learning-curve point.
Predefine the checkpoint schedule before inspecting the test scores.
The output intervals describe the requested comparisons; they supply no correction for multiple hypothesis tests.

## 8. Select hyperparameters with equal search budgets

The default matched settings define a fixed, untuned comparison.
For a tuned study, give every matched method the same three learning-rate candidates and tuning seeds.
The configuration files specify `0.00015`, `0.0003`, and `0.0006`.
Use training seeds `100,101,102` for these runs.

```bash
python make_plan.py --track matched --seeds 100,101,102 \
  --settings configs/lr_low.json --out plans/tune_low.jsonl
python make_plan.py --track matched --seeds 100,101,102 \
  --settings configs/lr_middle.json --out plans/tune_middle.jsonl
python make_plan.py --track matched --seeds 100,101,102 \
  --settings configs/lr_high.json --out plans/tune_high.jsonl
```

Train these plans in `tuning_runs`. Evaluate each run with the validation split:

```bash
python eval.py --plan plans/tune_low.jsonl --index 0 --root tuning_runs \
  --protocol common --clocks irregular --split validation --episodes 10
```

Apply that evaluation command to every tuning run and candidate.
Then freeze the final test plan:

```bash
python select_settings.py --plans plans/tune_low.jsonl plans/tune_middle.jsonl plans/tune_high.jsonl \
  --root tuning_runs --episodes 10 --test-seeds 0,1,2,3,4,5,6,7,8,9,10,11 \
  --out plans/matched_selected.jsonl
```

Selection uses the mean nominal irregular-clock validation time score for each method and task.
It requires identical tuning seeds, equal candidate counts, and disjoint final training seeds.
Register any larger search space and its compute budget before running it.
Published top configurations and fresh validation-selected configurations need separate provenance labels.

## 9. What the matched TPHT update establishes

The high-dimensional experiment replaces the Acrobot tensor grid with a batch of replay anchors \((s_i,a_i)\).
A frozen policy supplies one successor action \(a_i'\).
Positive nearest-neighbour weights interpolate its successor Q value from those anchors.
The code scales the observation and action blocks separately in this distance.

With the actor, anchors, successor samples, and weights fixed, the finite operator is

\[
 T_\pi z=b+\Gamma Pz,\qquad
 \Gamma_{ii}=(1-\mathrm{terminated}_i)e^{-\beta u_i},\qquad P\mathbf1=\mathbf1,
\]
\[
 b_i=R_i-\Gamma_{ii}\alpha W\log\pi(a_i'\mid s_i'),\qquad
 u_i=h_i/h_0,\quad \beta=-\log\gamma_{\rm native}.
\]

Here \(R_i\) is the discounted physics-substep reward integral in native control-time units.
\(W=\mathbb E[\int_0^{h/h_0}e^{-\beta t}dt]\) sets the entropy scale for the renewal decision clock.
The final clipped interval can have a different duration law.
The code keeps that observed duration for reward integration and discounting.

For this finite operator, let \(A=I-\Gamma P\), \(e(z)=Az-b\), and \(\bar\gamma=\max_i\Gamma_{ii}<1\).
The operator is a sup-norm contraction, and \(A\) is invertible.
The recovery function is explicit:

\[
 G(e)=A^{-1}(b+e),\qquad
 H_Q(z,\rho)=\tfrac12(\|e(z)\|_2^2+\|\rho\|_2^2),\qquad
 P_z=A^\top\rho.
\]

The auxiliary \(\rho\) is conjugate to the residual coordinate.
\(P_z\) is its conjugate momentum in the finite critic-value coordinate.
Both are solver variables. The robot's joint momenta have a separate physical meaning.

The residual phase follows

\[
 \dot e=-d e+\omega\rho,\qquad
 \dot\rho=-\omega e-d\rho.
\]

An implicit midpoint step contracts its storage when \(d>0\).
The code solves this two-coordinate phase exactly and recovers the critic target through \(G\).
The frozen operator is affine, so the inverse uses one dense linear solve.
The actor and the two neural critics then receive their usual single gradient updates.

With \(\rho_0=0\) and one phase step, the affine case has a useful equivalence:

\[
 z^+=z^\pi+c(z-z^\pi),\qquad z^\pi=A^{-1}b,
\]

where \(c\) is the diagonal coefficient of the midpoint phase map.
This identity explains the direct-solve baseline and prevents an advantage from inverse recovery being assigned solely to the pH interconnection.
The benchmark can show that a direct solve or ordinary Picard is cheaper.

`ct_picard` uses \(z^+=z+\eta(T_\pi z-z)\).
It is a CT-style finite-operator relaxation.
For an observed interval \(u_i\), its rate representation uses pseudo-step \(\tau_i=\eta u_i\).
The released CT-SAC method has its own single-critic target and target-network averaging.
It remains separately executable through the released track.

The corrected CT target in the matched track is

\[
 Q_{\rm fast}=r_{\rm end}+S(s)+
 \frac{(1-\mathrm{terminated})e^{-\beta u}S(s')-S(s)}{u},\qquad
 S(s)=\mathbb E[Q_{\rm target}(s,a)-\alpha\log\pi(a\mid s)].
\]

It bootstraps through time-limit truncations and applies the true-terminal mask to the successor term.
The released code applies its done mask to the whole correction.
Keep the method names distinct when reporting results.
This bundle uses first-order CT targets. A Richardson comparison needs measured half-interval transitions and a separate data-cost budget.

Each batch creates a new operator and resets \(\rho\) to zero.
The storage identity applies within that frozen solve.
Replay, actor updates, kernel projection, and neural fitting change the learning problem between solves.
Their errors require measurement.
The finite value bound is

\[
 \|z-z^\pi\|_\infty\le
 \frac{\|Az-b\|_\infty}{1-\bar\gamma}.
\]

Use the post-training audit to test it on actual frozen replay batches:

```bash
python audit_solver.py --run PATH_TO_MATCHED_RUN --batches 8 \
  --device cpu --out results/frozen_solver_audit.json
```

The audit reuses the same anchors, samples, and operator for all solvers.
Training logs also record neural fitting error, inverse residual, kernel projection defect, and nearest-anchor distance.
The projection defect compares kernel interpolation with the target network's successor prediction.
It supplies a consistency diagnostic for the sampled operator.
Additional model or simulator evidence is needed to bound error against the true physical control problem.
The finite storage theorem establishes no global convergence, passivity of the learned robot controller, or cross-robot generalization claim.

## 10. References and source record

- [CT-RL paper, arXiv:2602.14587v1](https://arxiv.org/abs/2602.14587v1), especially Sections 3–5 and Appendices B, F, and I.
- [CT-RL source repository](https://github.com/mpnguyen2/ct-rl). This bundle uses the supplied ZIP, with archive hash `152162e3633cd634530efb9cc007c7d014cb145a4bbcbdc93b4d141f6d299cbc`.
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control).
- [MuJoCo point Jacobian and force APIs](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).

The vendored source remains unchanged. Its original authors retain their rights.
The benchmark scripts add no claim of ownership over that source.
See `VALIDATION.md` for the checks performed during preparation.
