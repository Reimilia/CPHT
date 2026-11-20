# Pure CPHT: learn mechanical phase-space dynamics and compute feedback

Pure CPHT is an experimental actor-free method. It learns a physical Hamiltonian and a contact approximation, searches for a task-specific periodic motion, and computes six bounded motor commands through feedback. The executable method key is **pure_cpht**.

This name identifies the new mechanical-control experiment. The manuscript's CPHT HJB/2PBVP value iteration requires a separate implementation and theorem check. This code learns no value function or SAC actor/critic.

## What the uploaded videos establish

The CT-SAC clips show substantial travel through the course. The physical-feature SAC clips show little travel, with different drift under the two clocks. These observations motivate a new controller. Training logs and model/checkpoint diagnostics would be needed to assign a cause.

In v3.3, the three fitted physical coefficients supplied features to SAC. They imposed no kinematic constraint, physical-energy decrease rule, or passivity filter on the policy. Differences in features, reward/entropy scale, and critic target all remain possible explanations for the policy outcome. The optional raw-feature SAC and physical-feature CT-SAC controls still help separate these effects.

A calibrated physical model predicts the effect of an action. A task objective must also choose a useful motion. This revision supplies that missing connection explicitly:

1. Identify the physical differential equations from measured forces and motion.
2. Use those equations to search for a periodic motion that advances the robot.
3. Convert the motion into actuation through bounded feedback.
4. Use new interaction data to update the physical and contact models.

Positive actuator work is allowed. Locomotion consumes energy through the motor port. Physical damping is identified from data and included in the inverse feedback calculation.

## Run the new method

Run commands from the extracted TPHT_CT_RL_Benchmark folder. Activate the existing Torch environment and install requirements-physical-first.txt as needed. The runtime checks used Torch 2.9.1+cpu and EGL. The dependency list preserves the server's Torch/CUDA installation.

The new launcher defaults to CPU and dry run. Inspect its jobs first:

~~~bash
CASE=both STAGE=prepare bash scripts/run_pure_cpht.sh
CASE=both STAGE=train bash scripts/run_pure_cpht.sh
~~~

Run the experiment on your server:

~~~bash
export CASE=both
export TERRAIN=terrain
export METHODS=pure_cpht
export SEEDS=0
export MUJOCO_GL=egl

STAGE=prepare DRY_RUN=0 bash scripts/run_pure_cpht.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_pure_cpht.sh
~~~

The default options are in cases/pure_cpht.json. They allocate 4096 calibration decisions and 100000 total decisions, including calibration. Preparation fits physical and contact parameters. Training collects feedback-controlled motion and refits the models every 512 control decisions. Each motion plan uses a short learned-model rollout and a cross-entropy search over periodic-potential coefficients. There are zero actor, critic, Bellman, or finite value-inverse updates.

Use RESUME=1 to restore the last saved episode boundary. SIGINT, SIGTERM, and SIGUSR1 request a save at the next boundary. Intermediates are policy_*.pt; latest.pt also stores physical measurements for recovery. final.pt contains the learned plant, contact parameters, gait warm start, settings, and counters.

The default planner is computationally heavier than a neural actor. One initial default CPU plan took about 0.8 s in the local check. It runs once per 0.04 s of physical time, with cheaper feedback between plans. This is an offline experiment, with no real-time claim. DEVICE=cuda is available; CUDA performance and recovery remain untested here.

## Deploy feedback alone or continue motion replanning

First test the requested feedback-only deployment. It keeps the saved gait potential and physical/contact model fixed:

~~~bash
CASE=both METHODS=pure_cpht STAGE=eval DRY_RUN=0 DEVICE=cpu \
  FEEDBACK_ONLY=1 EPISODES=5 VIDEOS=1 \
  bash scripts/run_pure_cpht.sh
~~~

Then test receding-horizon feedback:

~~~bash
CASE=both METHODS=pure_cpht STAGE=eval DRY_RUN=0 DEVICE=cpu \
  FEEDBACK_ONLY=0 EPISODES=5 VIDEOS=1 \
  bash scripts/run_pure_cpht.sh
~~~

The latter recomputes the motion proposal from the current measurements. Its physical-model parameters stay frozen. Motion optimization and observer-state updates are control computations; deployment model-learning updates remain zero.

Evaluation writes task JSON, speed/progress PNGs, and headless MP4s for fixed and irregular clocks. Videos label Pure CPHT feedback or Pure CPHT MPC. Feedback-only files have a feedback suffix, so both modes can be inspected. The result key pure_cpht_feedback names this deployment ablation; its trained method remains pure_cpht.

Direct checkpoint evaluation is also available:

~~~bash
python eval_physical_fj.py --checkpoint-file /path/to/final.pt \
  --feedback-only --episodes 5 --video-episodes 1 \
  --clocks fixed,irregular --device cpu --gl egl
~~~

For F-to-J transfer with frozen physical parameters:

~~~bash
CASE=F CASE_TEST=J TERRAIN=terrain TERRAIN_TEST=steps \
  METHODS=pure_cpht COHORTS=layout FEEDBACK_ONLY=1 \
  STAGE=eval DRY_RUN=0 EPISODES=5 VIDEOS=1 \
  bash scripts/run_pure_cpht.sh
~~~

F uses fixed source course seed 1. J draws fresh courses. The terrain variants remain hills, friction, terrain, slopes, and steps. The task remains a 20 m path at 3 m/s with a 12 s deadline.

## Start with a short installation check

~~~bash
export CASE=both
export METHODS=pure_cpht
export OPTIONS=cases/pure_cpht_smoke.json
export CONFIG=cases/physical_first_smoke_task.json
export ARTIFACTS=artifacts/pure_smoke
export RUN_ROOT=runs/pure_smoke

STAGE=prepare DRY_RUN=0 bash scripts/run_pure_cpht.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_pure_cpht.sh
STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=1 VIDEOS=1 \
  FEEDBACK_ONLY=1 bash scripts/run_pure_cpht.sh
~~~

These settings use 64 calibration decisions, four subsequent control decisions, and a 0.08 s task deadline. They check learning, checkpoint recovery, and rendering. They supply no locomotion-performance result. Unset OPTIONS, CONFIG, ARTIFACTS, and RUN_ROOT before using the main defaults.

## Reproduce the bundled one-second feedback precheck

A small model JSON is supplied for an immediate check without training:

~~~bash
python preview_pure_cpht.py \
  --model-json prechecks/pure_cpht/feedback_precheck_model.json \
  --seconds 1 --out prechecks/pure_feedback.mp4
~~~

It contains the model from 64 calibration decisions and the gait warm start from four control interactions. Its task deadline is explicitly set to the ordinary 12 s value; the preview stops after one second. It starts on the initial flat pad, uses saved-gait feedback, and performs zero actor, critic, or model-learning updates. The remaining path can still be unfinished.

In the local check this controller advanced about 0.80 m and ended near 1.83 m/s without falling. Speed RMSE was about 2.29 m/s, including acceleration from rest. This establishes short actuation feasibility under a seeded periodic reference. Learning gains and stable-gait quality require zero-seed and unlearned-model controls plus longer runs. The preview and its metrics are in prechecks/pure_cpht/.

Use --checkpoint-file /path/to/final.pt for a trained Pure CPHT checkpoint. Add --replan to exercise online motion planning during the preview. The clip duration must stay within the saved task deadline.

To test motion discovery with zero initial Fourier amplitudes, set gait_seed_scale to 0 in the pure options. The default scale is 1. Compare zero-seed, seeded-reference, and optimized-reference runs before attributing locomotion gains to motion learning.

## Measurements and a matched CT-SAC baseline

The new preset uses the previous exact state and force-port readings plus the existing noisy local height/slope sensor. Both compared methods receive the same measurements when they use the same OPTIONS file.

The raw vector contains 45 entries: q (9), v (9), generalized constraint force (9), four task/clock scalars, and seven height/slope pairs. The four scalars are remaining distance/path length, remaining time/deadline, target speed/10, and requested hold/reference time. Each height/slope pair is measured at a declared horizontal offset from the torso. The offsets are -1, -0.5, 0, 0.5, 1, 2, and 3 m.

The controller reconstructs a local terrain scan from those measurements and the measured torso pose. Height/slope interpolation uses only this current scan. It receives no future terrain query, map ID, true friction, or material label. Calibration additionally uses acceleration, actuator torque, contact point/normal/force labels, and aggregate constraint forces. These are simulator-instrumented measurements.

For a fresh comparison with the same information and interaction budget:

~~~bash
CASE=both METHODS=ct_sac,pure_cpht OPTIONS=cases/pure_cpht.json \
  STAGE=prepare DRY_RUN=0 bash scripts/run_pure_cpht.sh
CASE=both METHODS=ct_sac,pure_cpht OPTIONS=cases/pure_cpht.json \
  STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_pure_cpht.sh
CASE=both METHODS=ct_sac,pure_cpht OPTIONS=cases/pure_cpht.json \
  STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 \
  bash scripts/run_pure_cpht.sh
~~~

The earlier uploaded CT-SAC videos used the previous input contract. They provide qualitative context. A matched numerical comparison needs fresh CT-SAC runs with these options. Existing v3.3 settings, artifact IDs, run IDs, and checkpoints remain supported when the new options are absent.

Use compare_physical_fj.py on one case/cohort/clock at a time. It checks matching calibration IDs, training seeds, actual interaction budgets, evaluation seeds, and course hashes. It can compare CT-SAC, Pure CPHT MPC, and Pure CPHT feedback reports. Report model-query counts and wall time alongside interaction count; the planning computation differs substantially.

## Learned mechanical phase space

The configuration manifold is locally a chart of $SE(2)\times(SO(2))^6$, restricted by joint limits. It has nine coordinates: root-x, root-z, torso pitch, and six joint angles. Use $q,v\in\mathbb R^9$ in this chart. The known topology, link geometry, joint axes, and motor gears come from the public robot XML.

The fitted parameters are seven link masses, seven planar rotational inertias, six armatures, six spring stiffnesses, and six damping coefficients. All 32 remain positive. The model constructs

$$
M_\theta(q)
=\sum_b m_b J_b(q)^\top J_b(q)
+\sum_b I_{b,y} a_ba_b^\top
+\operatorname{diag}(0,0,0,A_1,\ldots,A_6),
\tag{C1}
$$

where $J_b\in\mathbb R^{2\times9}$ is the link-center Jacobian and $a_b\in\mathbb R^9$ identifies its angular ancestors. These matrices follow from vectorized forward kinematics. Their articulated coupling is retained.

Define the learned momentum, potential, and Hamiltonian:

$$
p=M_\theta(q)v,\qquad
V_\theta(q)=-\sum_bm_b\,g^\top r_b(q)
+\tfrac12\sum_{j=1}^6K_j(q_{j+3}-q_{j+3}^{\,0})^2,
$$

$$
H_\theta(q,p)=\tfrac12p^\top M_\theta(q)^{-1}p+V_\theta(q).
\tag{C2}
$$

Thus $(q,p)\in T^*\mathcal Q$ has dimension 18. The momentum is inferred through the learned mass tensor. This is the latent mechanical lift used here. The scalar motion phase introduced below is a different variable.

The code uses known geometric coordinates and identifies the mechanical parameters. Learning an unknown observation-to-configuration encoder is a separate identification problem.

With $D_\theta=\operatorname{diag}(0,0,0,d_1,\ldots,d_6)$, the model is

$$
\dot q=\partial_pH_\theta,\qquad
\dot p=-\partial_qH_\theta-D_\theta\partial_pH_\theta
+Bu+\sum_cJ_c(q,r_c)^\top f_c+\tau_{\rm stop}.
\tag{C3}
$$

The corresponding input maps on phase space are $G_a=[0;B]\in\mathbb R^{18\times6}$ and $G_c(q,r_c)=[0;J_c(q,r_c)^\top]\in\mathbb R^{18\times2}$. For a rotating ancestor joint at $o_j$, the planar point-Jacobian column is $(r_{c,z}-o_{j,z},-(r_{c,x}-o_{j,x}))^\top$. Root translation columns are the two coordinate unit vectors. Joints outside the contact body's ancestor chain have zero columns. A back-foot port therefore directly forces its ancestor joints; the coupled inverse mass matrix can still change front-joint acceleration.

Here $B\in\mathbb R^{9\times6}$ has three zero root rows and gears 120, 90, 60, 90, 60, and 30 in its joint rows. Contact forces are planar normal/tangent forces. The 3D MuJoCo calibration measurements are projected into this planar model.

**Mechanical structure.** Positive parameters make (C1) positive definite for the supported tree, and $D_\theta$ positive semidefinite. The resulting power identity is

$$
\dot H_\theta
=v^\top Bu+\sum_c(J_cv)^\top f_c+v^\top\tau_{\rm stop}
-v^\top D_\theta v.
\tag{C4}
$$

*Proof.* Positive armatures control the six joint-velocity directions. Positive link rotational inertia controls root rotation, and positive mass controls root translation. Their sum gives a positive quadratic form. Apply the chain rule to (C3): the skew interconnection terms cancel and each port contributes force times its conjugate velocity.

Equation (C4) allows positive motor work. It describes the learned continuous plant. Underactuation, actuator bounds, contact approximation, and time stepping require separate checks for a controller or discrete-time certificate.

## Fit the differential form

The rigid-body inverse equations are linear in these physical parameters when q, v, and acceleration are measured:

$$
X(q,v,\dot v)\theta=\tau_{\rm act}+\tau_{\rm constraint}.
\tag{C5}
$$

The mass columns are $J_b^\top(J_b\dot v+a_b^{\rm bias}-g)$. The inertia columns are $a_b(a_b^\top\dot v)$. Remaining columns contain joint acceleration, spring displacement, and velocity.

The original three-scale calibration supplies an initial positive prior. The new fit uses all 32 columns with positive bounds and a small ridge term around that prior. Whole episodes form training and validation sets. New online episodes receive fixed validation roles; their labels are excluded from fitting when assigned to validation. A held-out relative mechanical force error above 5% rejects the update.

**Identification boundary.** With exact labels, full column rank, and zero ridge, a feasible true parameter is the unique solution of (C5). A positive ridge selects a unique regularized fit even when excitation leaves some combinations unidentified.

*Proof.* Full column rank makes the nullspace trivial. With positive ridge, the weighted least-squares Hessian has a positive diagonal addition, which gives strict convexity. This ensures a unique regularized minimizer; it provides no extra measurement information about unexcited parameter combinations.

At each model update the code compares estimated energy at the same measured q/v and records model_energy_recalibration_jump. This change belongs to the estimated model. Physical actuator work and within-step integration defects are reported separately.

## Learn the environment port and update its observation

The predictor represents robot capsules by 16 public endpoint candidates. It selects up to six nearby points and builds the current normal/tangent point Jacobian $J_c(q)$ from the measured local terrain. The radius lever arm is included. The active set and Jacobian change with motion.

For a frozen state and action, define

$$
a_{\rm free}=M_\theta^{-1}(Bu-c_\theta),\qquad
A_c=J_cM_\theta^{-1}J_c^\top,
$$

$$
R_c=\operatorname{diag}\!\left(\rho\,\operatorname{diag}(A_c)+10^{-5}\right).
\tag{C6}
$$

The desired normal acceleration is $-k\,g_c-b\,v_n$, where $g_c$ is signed contact gap. The desired tangent acceleration is $-b\,v_t$. The contact approximation solves

$$
\min_{f\in\mathcal C(\mu)}
\tfrac12 f^\top(A_c+R_c)f
-f^\top(a_{\rm ref}-J_ca_{\rm free}-a_c^{\rm bias}),
\quad
\mathcal C(\mu)=\{f_n\ge0,\ |f_t|\le\mu f_n\}.
\tag{C7}
$$

Inactive points have zero force. Parameters $k,b,\rho,\mu$ are learned from contact measurements. The normal fit uses the measured acceleration relation $a_n=-k g_c-bv_n-\rho(A_c)_{nn}f_n$. Sliding force ratios initialize friction.

The solver uses projected accelerated iterations and records its projected residual. The positive regularizer makes each frozen contact QP strictly convex. Finite iterations still have numerical error. This model approximates MuJoCo's contact response; it does not duplicate its solver.

A causal force observer compares the current measured generalized constraint force with the prediction under the previously held motor command. It solves a regularized point-force reconstruction with the same friction cones. A correction is retained only when it reduces current port error. Future model rollouts carry that correction on the same geometric points, with an exponential 0.04 s decay. Inactive future contacts suppress it.

This observer state absorbs some local contact-model and scan error. It reads the current force port and current scan. It does not refit model weights during evaluation.

## Learn a motion potential and compute actuation

The task search chooses 19 coefficients: six offsets, six sine coefficients, six cosine coefficients, and one frequency. They define a periodic joint reference

$$
q_\psi(\varphi)=c_\psi+A_\psi\sin\varphi+B_\psi\cos\varphi,
\qquad
\dot\varphi=2\pi\nu_\psi.
\tag{C8}
$$

All coefficients remain free within declared bounds. The initial amplitudes seed exploration; no measured gait demonstration is used. Joint references respect the public joint limits.

This reference is the minimum of a time-dependent joint potential

$$
U_\psi(q_j,\varphi)
=\tfrac12(q_j-q_\psi(\varphi))^\top W_{\rm ref}(q_j-q_\psi(\varphi)).
\tag{C9}
$$

Here $W_{\rm ref}$ is a fixed positive reference metric. The code represents this potential through its Fourier center. $U_\psi$ is a reference-space objective; $H_\theta$ is the mechanical energy.

A cross-entropy search optimizes these coefficients through short rollouts of (C3), (C6), and (C7). The cost includes speed error, torso clearance, pitch, motor effort, progress shortfall at the target speed, and scan extrapolation. Failed predicted trajectories retain a failure penalty. The default horizon is approximately 0.2 s. The first command is held for the measured requested duration; later predicted commands use the declared 0.01 s planning clock. The integrator uses substeps no larger than 0.005 s and keeps each command fixed within its hold. It reads no future sampling times. Actual task termination can shorten the first hold.

The reference supplies desired joint acceleration

$$
a_j^\star=\ddot q_\psi+K_P(q_\psi-q_j)+K_D(\dot q_\psi-v_j).
\tag{C10}
$$

Root speed, clearance, and pitch feedback supply three additional desired accelerations. At the current measured force port $\tau_e$, let

$$
a_0=M_\theta^{-1}(\tau_e-c_\theta),\qquad
A_u=M_\theta^{-1}B.
$$

Compute motor commands from

$$
u^\star=\arg\min_{u\in[-1,1]^6}
\tfrac12\|W^{1/2}(A_uu+a_0-a^\star)\|_2^2
+\tfrac{\lambda_u}{2}\|u\|_2^2.
\tag{C11}
$$

The implementation uses a small batched primal active-set solver. It fixes inputs at their active bounds, solves the remaining quadratic, takes a feasible step to the next blocking bound, and releases a bound with a wrong-sign KKT multiplier. The iteration cap is 16. Its logged projected residual measures the remaining optimality defect.

**Bounded feedback.** At each fixed state, $\lambda_u>0$ gives (C11) a unique minimizer. Each feasible active-set step respects motor bounds. The reported residual is zero at the box-constrained optimum, up to numerical precision.

*Proof.* The Hessian is $A_u^\top W A_u+\lambda_u I\succ0$. The input box is compact and convex, which gives existence and uniqueness. A step stops at the first blocking bound, so feasibility is preserved. The condition $u=\Pi_{[-1,1]^6}(u-\nabla\ell(u))$ is equivalent to the convex KKT condition. The code records the defect of this condition and caps its iterations.

There are six actuator directions. The controller retains the three floating root equations, the non-diagonal mass tensor, and the changing contact forces. The reference potential proposes motion. Exact closed-loop Hamiltonian reshaping or limit-cycle stability would require further matching and stability arguments.

In feedback-only deployment, the saved coefficients in (C8) stay fixed and (C11) supplies actuation. Receding-horizon deployment updates (C8) through the local task search. In both modes the learned physical/contact parameters stay fixed during evaluation.

## Algorithm sketch

~~~text
Input: case F/J, public robot geometry/gears, task, measured sensors,
       calibration budget, total interaction budget, independent seeds.
Prepare:
    Collect random-action episodes and physics-step force/acceleration labels.
    Save measurements and split whole episodes into fixed roles.
    Fit the three-scale prior, then 32 positive mechanical parameters, (C5).
    Fit four contact parameters and measure contact/rollout error, (C6)-(C7).
    Initialize the latent periodic potential coefficients, (C8)-(C9).
Interact and learn:
    Observe q, v, current force port, task, hold, and local terrain scan.
    Infer p = M_theta(q)v and update the causal contact-force observer.
    At each planning event:
        Search periodic-potential coefficients through learned-model rollouts.
    Compute six bounded motor commands with feedback, (C10)-(C11).
    Execute the held command and record actual duration and physical labels.
    Periodically refit mechanics/contact parameters on training episodes.
    Check held-out errors; retain previous contact parameters after regression.
    Record model-energy recalibration jumps separately from physical work.
    Save the plant, contact model, gait warm start, data, counters, and episode.
Deploy:
    Freeze physical/contact parameters.
    Choose saved-gait feedback or continued receding-horizon feedback.
    Advance the motion phase by actual physical duration.
    Evaluate paired courses/clocks; save task and prediction metrics plus MP4.
~~~

## What to examine before increasing the budget

Mechanical calibration and locomotion are separate validation targets. Check held-out mechanical force/power error, contact-port prediction error, observer correction error, solver residuals, and held-control state-prediction error. A small mechanical force residual can coexist with a large contact-model error.

For the task, report completion, progress, speed RMSE, time within the speed band, falls, timeouts, motor saturation, and actuator work. Compare feedback-only and replanned deployment to determine how much the online motion search contributes. Use independent training seeds for uncertainty; repeated videos alone provide no training-seed uncertainty estimate.

The first improvements to investigate are better contact closure, richer local environment inference, and a planning horizon that covers the useful motion cycle. The current capsules/endpoints and interpolated scan approximate steps and contact transitions. A global friction coefficient averages a changing environment. Positive model parameters give mechanical structure; predictive contact accuracy remains empirical.

The default 0.2 s horizon is shorter than a full cycle at 2.3 Hz. For a longer-horizon experiment, cases/pure_cpht_cycle.json uses a 0.5 s horizon and 30000 total decisions. It costs more per plan. Check contact prediction before expanding the horizon, because long rollouts can amplify model error. Stable learned limit cycles would also benefit from a periodic boundary/return-map objective and a separate stability check.

Unknown observation coordinates, noisy acceleration/force labels, arbitrary robot topologies, contact uncertainty ensembles, and the manuscript's value-space CPHT solver remain further work. This first method uses a physically anchored latent momentum and an explicit task-driven motion layer.

Implementation references: the [MuJoCo computation chapter](https://mujoco.readthedocs.io/en/stable/computation/index.html) describes the joint-space and contact-force equations. [Tedrake's contact planning chapter](https://underactuated.mit.edu/contact.html) explains the role of contact modes and periodic motion planning. The new algorithm and its experimental boundaries are specified above.
