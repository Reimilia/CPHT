# Physical calibration first: a small cheetah experiment

For the new actor-free Pure CPHT method, use [PURE_CPHT.md](PURE_CPHT.md). This guide describes the retained SAC feature experiments.

This pipeline tests whether a calibrated mechanical representation helps SAC on changing contact environments. It learns three positive physical coefficients, freezes them, and trains a policy. It uses the CT terrain environments and finite path task from v3.2.1.

The executable method key `cpht_sac` selects this physical-first pilot. Its policy and critic use ordinary interval SAC updates. This pilot tests physical features. The manuscript's CPHT Hamiltonian–Picard/HJB value solver remains a separate experiment. A trained policy is needed to measure any control benefit.

## Run the pilot

Extract the ZIP and run the commands from `TPHT_CT_RL_Benchmark/`. Use a new run directory for these new observations. Earlier checkpoints keep their earlier interfaces and use `eval_fj.py`.

Activate your existing Torch environment. Install the small simulator, plotting, and video dependency list as needed:

```bash
python -m pip install -r requirements-physical-first.txt
```

This list preserves the installed Torch/CUDA package. Checks used Python 3.12, Torch 2.9.1+cpu, MuJoCo 3.17.0, and dm-control 1.0.50. Configure EGL on a headless server. OSMesa is an alternative when it is installed.

Inspect the plan first. Every launcher command defaults to `DRY_RUN=1`:

```bash
CASE=both METHODS=ct_sac,cpht_sac SEEDS=0 STAGE=prepare \
  bash scripts/run_physical_first.sh
```

Run shared physical calibration and train the two selected methods on your server:

```bash
export CASE=both
export TERRAIN=terrain
export METHODS=ct_sac,cpht_sac
export SEEDS=0
export MUJOCO_GL=egl

STAGE=prepare DRY_RUN=0 bash scripts/run_physical_first.sh
STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_physical_first.sh
```

Preparation uses 2048 control decisions per case/seed. It performs a three-variable least-squares fit and zero policy updates. The default total budget is 1,000,000 decisions, including calibration. Training uses one actor and critic update per policy decision. There is no observer fit, reward-model fit, node bank, or finite value inverse in this pipeline.

Choose only one method with `METHODS=cpht_sac` or `METHODS=ct_sac`. Use `SEEDS=0,1,2` for three independent replicates. Preparation is shared by methods for each seed. Keep the same settings at prepare, train, and evaluation stages. `RESUME=1` restores `latest.pt` at its saved episode boundary. `SIGINT`, `SIGTERM`, and `SIGUSR1` request a save at the next episode boundary.

The terrain choices are `hills`, `friction`, `terrain`, `slopes`, and `steps`. The default `terrain` combines elevation and friction patches. F uses fixed course seed 1. J draws a fresh course at each episode. The CT terrain backend keeps the stock robot and contact material law. The earlier mesh/material backend can be selected through a complete `CONFIG` JSON with `contact_backend=mesh`, `contact_curriculum=false`, and `contact_robot_variation=false`.

`OPTIONS=learning.json` changes physical-first settings such as `hidden`, `batch`, `lr`, `capacity`, `calibration_episode_steps`, and the budgets. `CONFIG=task.json` selects environment/task overrides. `TOTAL_DECISIONS` and `CALIBRATION_DECISIONS` override their learning settings. The custom task JSON should contain the intended physical backend and profile, as shown in `cases/physical_first_smoke_task.json`.

## Evaluate checkpoints and save headless video

```bash
CASE=both TERRAIN=terrain METHODS=ct_sac,cpht_sac SEEDS=0 \
  STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 \
  bash scripts/run_physical_first.sh
```

The launcher finds each `final.pt`. It evaluates paired fixed and irregular clocks. Metrics and speed/progress PNGs go to the run's `evaluations/` directory. MP4s and recording sidecars go to `videos/`. Recording and metrics use the same trajectory. `VIDEOS=0` disables video. Set `OVERWRITE_VIDEOS=1` to replace an existing recording.

Evaluate a checkpoint directly, including a saved intermediate `policy_*.pt`:

```bash
python eval_physical_fj.py --checkpoint-file /path/to/final.pt \
  --episodes 5 --clocks fixed,irregular --cohorts iid,layout \
  --video-episodes 1 --device cpu --gl egl
```

For frozen cross-environment transfer, keep `CASE` and `TERRAIN` equal to the training values. Select the test environment with separate options:

```bash
# An F-trained policy on fresh J step courses, with frozen calibration.
CASE=F TERRAIN=terrain CASE_TEST=J TERRAIN_TEST=steps \
  METHODS=ct_sac,cpht_sac SEEDS=0 COHORTS=layout \
  STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 \
  bash scripts/run_physical_first.sh
```

The direct equivalents are `--case-test J --terrain-test steps`. A terrain transfer changes the elevation/friction profile. Calibration, normalization, actor weights, and task cost remain frozen. `cohort=task` selects held-out target speeds in J. `cohort=layout` uses a separate course seed stream. F retains its fixed course; use F-to-J transfer to test new layouts after F training.

Compare one test setting and clock at a time:

```bash
python compare_physical_fj.py --evaluations \
  /path/to/ct_sac/evaluations/final_FtoF_terrain_iid_fixed_n5.json \
  /path/to/cpht_sac/evaluations/final_FtoF_terrain_iid_fixed_n5.json \
  --out reports/F_fixed.json --plot reports/F_fixed.png
```

The comparison checks matching calibration identities, training seeds, actual decision budgets, evaluation seeds, and course hashes. Each independent training seed is one replicate. A single seed gives a pilot result. Fix the evaluation protocol before training; this small pipeline uses a fixed final budget rather than automatic checkpoint selection on test data.

## A short installation check

The supplied smoke settings collect 64 calibration decisions and perform 16 policy updates per selected method. The episode deadline is 0.08 s. These settings check the executable path and supply no running-performance evidence.

```bash
export CASE=F
export METHODS=ct_sac,cpht_sac
export OPTIONS=cases/physical_first_smoke.json
export CONFIG=cases/physical_first_smoke_task.json
export ARTIFACTS=artifacts/physical_smoke
export RUN_ROOT=runs/physical_smoke

STAGE=prepare DRY_RUN=0 bash scripts/run_physical_first.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_physical_first.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=1 VIDEOS=1 \
  CLOCKS=fixed bash scripts/run_physical_first.sh
```

Unset these smoke variables before the main experiment. The ordinary defaults use the full path and deadline.

## What is learned

Let $q\in\mathbb R^9$ contain the floating planar root and six joint coordinates. Let $v=\dot q$. The public cheetah XML defines a nominal articulated mass function $M_0(q)$, gravitational potential $V_{g0}(q)$, spring potential $V_{s0}(q)$, and joint damping $D_0\succeq0$. The nominal template has deliberately biased mass, stiffness, and damping scales: 0.8, 1.2, and 1.4.

The calibration learns $\theta=(a,k,d)\in[0.05,20]^3$:

$$
M_\theta(q)=aM_0(q),\quad
V_\theta(q)=aV_{g0}(q)+kV_{s0}(q),\quad
D_\theta=dD_0.
\tag{P1}
$$

The conjugate mechanical momentum and physical Hamiltonian are

$$
p=M_\theta(q)v,\qquad
H_\theta(q,p)=\tfrac12p^\top M_\theta(q)^{-1}p+V_\theta(q).
\tag{P2}
$$

The mass function retains all articulated couplings from the XML. The fitted coefficients correct a restricted parameter family. General per-link inertias, new robot topologies, and a freely learned Hamiltonian require a larger model.

For the six cheetah motors, $B\in\mathbb R^{9\times6}$ has three zero root rows and joint gears (120,90,60,90,60,30). Thus $\tau_{\rm act}=Bu$. The pH equations are

$$
\begin{bmatrix}\dot q\\\dot p\end{bmatrix}
=
\left[
\begin{pmatrix}0&I\\-I&0\end{pmatrix}
-
\begin{pmatrix}0&0\\0&D_\theta\end{pmatrix}
\right]\nabla H_\theta
+
\underbrace{\begin{bmatrix}0\\B\end{bmatrix}}_{G_a}u
+
\sum_c\underbrace{\begin{bmatrix}0\\J_c(q,r_c)^\top\end{bmatrix}}_{G_c(q,r_c)}f_c
+
\begin{bmatrix}0\\\tau_{\rm other}\end{bmatrix}.
\tag{P3}
$$

Here $r_c$ is the active contact point, $f_c\in\mathbb R^3$ is its world force, and $J_c\in\mathbb R^{3\times9}$ maps joint velocities to point velocities. `mj_jac` constructs this point Jacobian. `mj_contactForce` supplies contact-frame force; the code rotates it to world coordinates and applies the body-order sign. Joint-limit reactions are included in $\tau_{\rm other}$. The policy receives the aggregate generalized constraint-force port.

The active contacts and $J_c(q,r_c)$ change with motion. A back-foot force has zero direct generalized force on front-leg joints. Its acceleration response is

$$
\Delta\dot v=M_\theta(q)^{-1}J_c(q,r_c)^\top\Delta f_c.
\tag{P4}
$$

The coupled inverse mass matrix can produce nonzero front-leg accelerations. `contact_examples.json` records actual points, Jacobians, forces, direct generalized forces, and calibrated acceleration responses from acquisition. Each record stores the matching $q$ and physical sample time.

For example, the recorded F sample has a back-foot world force approximately $(116.96,0,292.39)$ N. Its generalized force, in root-x/root-z/torso-angle/back-joints/front-joints order, is

$$
\tau_c\approx(116.96,292.39,104.93,-43.88,29.59,-42.18,0,0,0)^\top.
$$

The translational entries have force units; the angular entries have torque units. Equation (P4) gives front-joint acceleration contributions approximately $(-21.53,-9.30,-1.98)$ rad/s². This calculation isolates the back-foot port while holding all other generalized forces fixed. Re-solving contact after a new applied force can also change the other reaction forces.

## Measurements and fitting

Calibration uses exact simulator measurements of $q,v,\dot v,\tau_{\rm act}$, and $\tau_{\rm constraint}$. It calls `mj_forward` on a copied `MjData` at each physics substep. This produces mutually consistent instantaneous labels and preserves the simulated state and warm start. It uses continuous-time acceleration labels. MuJoCo's integrated finite differences include integration error and implicit damping effects, so they are a different measurement model.

Write $c_0(q,v)$ for nominal Coriolis, centrifugal, and gravitational forces. Each physical sample supplies

$$
X_i=
\begin{bmatrix}
M_0(q_i)\dot v_i+c_0(q_i,v_i)&
\nabla V_{s0}(q_i)&
D_0v_i
\end{bmatrix},\qquad
y_i=\tau_{{\rm act},i}+\tau_{{\rm constraint},i}.
\tag{P5}
$$

Calibration minimizes $\sum_i\|X_i\theta-y_i\|_2^2$ over the positive box in (P1). It scales the three design columns, checks rank, and uses bounded least squares. Whole episodes form the held-out set. Normalization and policy warm-start replay use training episodes only. A held-out relative force RMSE above 5% stops admission. Raw data are saved before fitting, so a failed fit can be inspected and retried without new acquisition.

For the current noiseless stock robot, the expected coefficients are (a=1/0.8=1.25), $k=1/1.2\approx0.833333$, and $d=1/1.4\approx0.714286$. This check is a supervised physical calibration around a known XML model. Measured environmental reactions are required; they carry the effects of terrain and friction. The calibration does not identify an unknown friction law from partial observations.

**Identifiability.** Suppose the chosen template is exact, the measured forces and accelerations are consistent, and the stacked $X$ has column rank three. If the true positive coefficients lie in the declared box, they are the unique least-squares minimizer.

*Proof.* With noiseless data, $y=X\theta_\star$, so the true coefficients attain zero loss. Any other zero-loss coefficient satisfies $X(\theta-\theta_\star)=0$. Full column rank gives $\theta=\theta_\star$. The Hessian $2X^\top X$ is positive definite, which also gives strict convexity on the box.

**Physical power balance.** Equations (P1)–(P3) imply

$$
\dot H_\theta
=v^\top\tau_{\rm act}
+\sum_c(J_cv)^\top f_c
+v^\top\tau_{\rm other}
-v^\top D_\theta v.
\tag{P6}
$$

*Proof.* Apply the chain rule to (P3). The skew interconnection contributes zero. The lower dissipative block contributes $-v^\top D_\theta v$, since $\partial_pH_\theta=v$. Each force port contributes its conjugate force–velocity product. Positivity of $a,k,d$ preserves the mass and damping signs.

This is a continuous-time structural identity. It gives no closed-loop stability certificate for an arbitrary SAC policy or a discrete-time passivity certificate for MuJoCo's Euler integrator. An independent finite-difference energy-gradient test checks (P6) in the regression suite.

## Inputs and update rules

Both policies receive the same causal measured information: $q$ (9), $v$ (9), generalized constraint force (9), and four task/clock scalars. The four scalars are remaining distance/path length, remaining time/deadline, target speed/10, and requested hold/reference time. This gives 31 raw features. The force reading at a decision uses the previously held control. Acceleration labels are used only during physical calibration.

The physical representation replaces $v$ with calibrated $p$. It appends kinetic energy, total potential, environment-port power, and joint-damping power. This gives 35 features. Scaling is frozen from training calibration episodes. Task/clock scalars retain their declared dimensionless scales. Neither policy receives terrain maps, friction labels, or future contact forces.

Four method keys support interpretation:

- `ct_sac`: raw features and the source-style CT-SAC target.
- `cpht_sac`: physical features and the measured-interval SAC target.
- `sac`: raw features and the same measured-interval target.
- `ct_sac_ph`: physical features and the same CT-SAC target.

The two default methods provide the requested pilot comparison. They change both features and target rules. The optional controls isolate each factor. In particular, compare `cpht_sac` against `sac` to test physical features under the same update rule, and against `ct_sac_ph` to test the update rule under the same features.

For a measured hold $h_k$, let $\bar h_k=h_k/h_{\rm ref}$ and $\gamma_k=\gamma_{\rm ref}^{\bar h_k}$. Let $R_k$ be the measured discounted interval reward, including the path event. Let $d_k$ mark either an actual terminal event or the finite task deadline. Let (Q^-) denote the smaller target critic.

The interval method uses

$$
y_k=R_k+\gamma_k(1-d_k)
\left[Q^-(s_{k+1},a')-
\alpha\frac{h_{k+1}^{\rm requested}}{h_{\rm ref}}
\log\pi(a'\mid s_{k+1})\right].
\tag{P7}
$$

Its actor loss is the mean of $\alpha(h_k^{\rm requested}/h_{\rm ref})\log\pi(a\mid s_k)-\min_jQ_j(s_k,a)$. The requested hold is known when the action is selected; the actual hold can end earlier at a terminal event.

The CT adapter follows the released `algorithms/ct_sac.py` target form. Put $V^-(s)=\mathbb E_a[Q^-(s,a)-\alpha\log\pi(a\mid s)]$, estimated with one policy sample. On a nonterminal interval it uses

$$
y_k^{\rm CT}=r_k^{\rm src}+V^-(s_k)
+\frac{\gamma_kV^-(s_{k+1})-V^-(s_k)}{\bar h_k+10^{-8}}.
\tag{P8}
$$

Its actor loss uses $\alpha\log\pi-\min_jQ_j$. At finite task termination both adapters use (y_k=R_k), which includes the terminal path event and suppresses bootstrapping. The terminal override, finite path reward, added measurements, and common architecture are declared benchmark adaptations. This is a source-style CT baseline on a new task, rather than an exact replay of a published score.

The implementation fits neural critics to (P7) or (P8), then updates the actor and target critics. It logs value/target magnitude and loss. It performs zero finite residual-recovery solves. Freeze calibration throughout these RL updates and throughout evaluation.

## Algorithm sketch

```text
Input: case F or J, public robot template, terrain/task settings,
       calibration budget Ncal, total budget N, methods, independent seeds.
For each seed and case:
    Acquire Ncal random-action control transitions in short episodes.
    At every physics substep, record the consistent labels in (P5).
    Save acquisition and contact/Jacobian examples.
    Split whole episodes into calibration training and validation sets.
    Fit positive (a,k,d); check rank and held-out force/power errors.
    Freeze the physical model and training-only normalization.
    For each selected method:
        Initialize actor, twin critics, and shared measured replay.
        Until the total interaction count reaches N:
            Observe causal state, force port, task, and requested hold.
            Form raw or physical features; sample a motor command.
            Execute it and store measured reward, duration, and successor.
            Update critics with (P7) or (P8); update actor and targets.
            Save weights and boundary recovery checkpoints.
        Freeze weights; evaluate paired seeds, terrain layouts, and clocks.
        Save task metrics, speed/progress plots, and optional MP4s.
Compare independent training seeds using matched calibration and test courses.
```

The default task is a 20 m path at 3 m/s with a 12 s deadline. Reward penalizes speed error, effort, elapsed time, and backward travel; it includes goal and failure events. Report completion rate first, then speed RMSE, relative speed RMSE, time within the 10% speed band, progress, falls, and timeouts. Completion time is conditional on success. Acceleration from rest contributes to speed error. Zero completion after a tiny smoke run has no value as an algorithm ranking.

## Why the earlier inverse failed

The legacy `fjbench/finite.py` solver evolves a critic residual $e=Q-\mathcal UQ$ and a solver momentum $\rho$. Its energy is $\mathcal H_{\rm solver}=\tfrac12\|e\|^2+\tfrac12\|\rho\|^2$. This differs in state, units, and purpose from the mechanical $H_\theta(q,p)$ in (P2).

Recovery forms $A=I-\operatorname{mean}_k(\gamma_{ik}P_{ik\ell})$ and solves for the soft value. The stochastic rows give $\|A^{-1}\|_\infty\le(1-\gamma_{\max})^{-1}$. Close-to-one discounts and large rewards/Q values can expose rounding in the absolute recovery test. The reproduced stress fixture has an absolute defect about $6.3\times10^{-8}$ and normalized defect about $7.7\times10^{-17}$. Its old absolute (10^{-8}) gate fails.

The included legacy correction uses float64 recovery, up to two refinement solves, and the gate

$$
\delta_\infty\le a_{\rm tol}+r_{\rm tol}
\max(1,\|Q\|_\infty,\|\mathcal UQ\|_\infty,\|e_{\rm requested}\|_\infty),
\tag{P9}
$$

with defaults $a_{\rm tol}=10^{-8}$, $r_{\rm tol}=10^{-12}$. Diagnostics retain the absolute defect, allowed defect, normalized defect, discount gap, and a conditioning bound. A failed recovery rejects the critic block before actor/critic updates. Nonfinite inputs fail explicitly. Large learned reward/Q drift remains a possible cause in a particular training run; its checkpoint diagnostics are needed to assess that run.

To patch only recovery in an existing project, with backups and unchanged old config/preparation identities:

```bash
python apply_finite_fix.py --target /path/to/TPHT_CT_RL_Benchmark_v2
python diagnose_finite_fj.py --checkpoint-file /path/to/old/latest.pt \
  --out reports/old_recovery.json
```

The diagnostic samples one frozen block from the checkpoint. Its action quadrature can differ from the originally failing block. The patch preserves the existing calibration data; a full new installation uses new run identities for the added numerical settings. The separate physical-first pipeline requires fresh physical calibration and policies.

## Scope of this first experiment

The pilot uses one known cheetah topology, a three-parameter physical family, exact simulator state, and instrumented constraint forces. Environmental elevation, friction, and contact response enter through the measured port. Robot parameter transfer, noisy or partial measurements, unseen external forces, online calibration, and learned contact-force estimation require further models and tests. Control improvement and terrain generalization remain empirical questions for the server runs.

Useful implementation references are the [MuJoCo computation chapter](https://mujoco.readthedocs.io/en/stable/computation/index.html), [point Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac), and the hash-checked CT-RL source in `vendor/ct-rl-source.zip`. The matched physical terrain is documented in `TERRAIN_ALIGNMENT.md`.
