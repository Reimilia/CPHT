# MuJoCo reference control and transition validation

This extension compares the supplied `sensor.zip` pure-CPHT checkpoints with local MuJoCo control and MuJoCo shooting. It runs evaluation and numerical audits. It performs zero policy training and zero parameter refits. Existing sensing, CT-SAC, CPHT-SAC, pure-CPHT, and differentiable-comparison experiments remain available.

## What MuJoCo solves

MuJoCo supplies forward dynamics, inverse dynamics, contact reactions, and numerical time integration for a specified physical model. A locomotion controller must additionally specify a task objective, a motion representation or action sequence, and a control update. A converged contact solve does not imply that the cheetah will finish its course.

For this benchmark, use a local coordinate chart of

\[
q\in SE(2)\times (SO(2))^6,\quad v\in\mathbb R^9,\quad a\in[-1,1]^6.
\]

The chart is `rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot`. `rooty` is the pitch angle. The first two generalized coordinates have length units, and the other seven have angle units. The six motors have gears

\[
B=\begin{bmatrix}0_{3\times6}\\\operatorname{diag}(120,90,60,90,60,30)\end{bmatrix}.
\]

For the direct motors in this model, generalized motor force is \(Ba\). The backend checks this identity before using the local fit. All physical rollouts retain the six actuator bounds. The floating base contributes three unactuated coordinates.

Write the dynamics as

\[
M_\theta(q)\dot v=Ba+\tau_c(q,v,a;\mathcal E)+\tau_{\rm ext}-c_\theta(q,v),\qquad \dot q=v.
\]

Here `c = qfrc_bias - qfrc_passive`; it includes gravity, velocity-dependent terms, springs, and damping with the corresponding signs. The environmental factor \(\mathcal E\) includes terrain geometry, contact material/friction, and any external load or force. `qfrc_constraint` also includes joint-limit and other constraint contributions. It is an aggregate generalized force, rather than a six-motor command or an independently actuated contact force.

## The five controls

1. `learned_feedback` uses the saved periodic gait, the learned mechanical model, the measured current constraint port, and the causal environment estimate. It performs bounded feedback without replanning.
2. `mujoco_ls` uses the same saved gait and target-acceleration construction, with exact current MuJoCo mechanics. It freezes the current constraint-force port while fitting a bounded command.
3. `mujoco_fd_ls` locally differentiates the full MuJoCo acceleration response, including the contact solve, with respect to each motor command. It fits the resulting local model within a trust box.
4. `learned_mpc` replans the periodic gait using the supplied learned pH/contact model and estimated terrain. The comparison uses a smaller, explicitly recorded search budget than the uploaded training run.
5. `mujoco_mpc` searches the same periodic-gait parameter space using actual MuJoCo contact dynamics in copied simulation states. Its feedback also uses exact mechanics. This is a privileged model reference.

The first three controls isolate whether changing current mechanics/contact response helps a fixed motion reference. The final two assess receding-horizon motion selection. Their results concern the specified controller families and compute budgets.

### Bounded current-state least squares

At current command \(a_0\), evaluate the full acceleration \(f_0\) and mass matrix. Define

\[
A=M^{-1}B,\qquad b=f_0-Aa_0.
\]

The frozen-port model is \(\dot v(a)\approx Aa+b\). In particular, changing \(a\) does not update contact forces inside this approximation. The fitted control solves

\[
\min_{-1\le a\le1}
\|W^{1/2}(Aa+b-d)\|_2^2+\lambda\|a\|_2^2,
\quad W=\operatorname{diag}(1,4,4,.15,.15,.15,.15,.15,.15).
\]

The implementation uses bounded linear least squares with an augmented matrix, rather than clipping an unconstrained inverse. It logs projected-gradient residual, weighted tracking residual, action saturation, rank, and full-contact acceleration after applying the selected command. Coordinate weights are numerical scales on acceleration expressed in m/s² and rad/s²; they are not an invariant kinetic-energy norm.

The desired acceleration \(d\) follows the existing controller: COM-speed error drives the root-x acceleration target, torso-clearance error drives root-z, pitch has PD feedback, and the six actuated joints track a periodic reference with feedforward acceleration and PD correction. This is a heuristic acceleration target for an underactuated articulated system. Arbitrary nine-dimensional targets need not be attainable.

### Contact-aware local least squares

Let \(F(a)\) be acceleration returned by a fresh `mj_forward` at the same current state. Compute

\[
J_i\approx\frac{F(a_0+\epsilon_i e_i)-F(a_0-\epsilon_i e_i)}{2\epsilon_i},
\qquad F(a)\approx F(a_0)+J(a-a_0).
\]

The implementation uses \(\epsilon=10^{-3}\), with one-sided differences at actuator bounds, and intersects the control box with \(\|a-a_0\|_\infty\le .5\). Each query starts with the same copied warm-start state. The full-contact acceleration at the chosen command is evaluated again and compared with the linear prediction. Contact changes can invalidate the local approximation; the diagnostic remains visible in the trace.

### Model-based shooting

The saved gait has 19 coefficients: six offsets, six sine coefficients, six cosine coefficients, and one frequency. It defines joint reference position, velocity, and acceleration. The existing controller clips joint references to joint limits and zeros their derivatives when clipping is active. This restriction is shared by the reference runs.

The checked comparison uses eight CEM candidates, three elites, one iteration, 20 control intervals, and replanning every .1 s. The current held duration is disclosed; later predicted holds are .01 s. The nominal horizon is .2 s, with small variation under an irregular first hold. Coefficients start from the supplied checkpoint gait; the random seed and coefficient bounds are shared. There is no task-policy or physical-parameter update during deployment.

The learned predictor retains its existing semi-implicit integration and contact closure. Native prediction uses MuJoCo's .002 s physics step. Native feedback uses the bounded current-state fit. Speed, clearance, pitch, effort, progress, and invalid-state penalties follow the existing planner weights. The native scan-coverage penalty uses the torso; the learned planner uses its capsule points. Thus the MPC comparison changes feedback mechanics, contact dynamics, integration, and that coverage penalty. It is a practical reference comparison and does not isolate a single model component. Elapsed wall time and predictive work are recorded.

All native candidates use copied `MjData`. The compiled `MjModel` is read without mutation during a planning call. Planning never invokes the real environment's sensor, reward, clock-RNG, or step callbacks. The input/transition audit verifies that action selection leaves the live integration state unchanged. Native MPC has full current terrain/friction information in its model, including terrain beyond the inferred scan. No future intervention schedule is revealed. Its privileged access must be kept explicit in any comparison with a learned or sensor-limited controller.

## How to run

Install `requirements.txt`. The local validation used MuJoCo 3.17.0 and dm-control 1.0.50. Extract `sensor.zip`; `SENSOR_ROOT` must point to the directory containing `F/` and `J/`.

The simplest route is the `verify` stage of the experiment launcher. It replays the recorded transitions, runs `tools/paper_checks.py`, and, when `SENSOR_ROOT` is set, the full reference comparison below into `runs/<scale>/mujoco_reference`:

```bash
SENSOR_ROOT=/absolute/path/to/sensor ./run_experiments.sh smoke verify
```

For finer control, call the module directly:

```bash
export SENSOR_ROOT=/absolute/path/to/sensor
export MUJOCO_GL=egl

# Full 20 m / 12 s paired episodes, both cases and both clocks; no training.
python -m mujoco_reference compare --sensor-root "$SENSOR_ROOT" --output outputs/reference

# Fast first check: exact local controls and saved learned feedback.
python -m mujoco_reference compare --sensor-root "$SENSOR_ROOT" --output outputs/reference_local \
  --methods learned_feedback mujoco_ls mujoco_fd_ls --clocks fixed

# Generate real terrain-state transitions with five .002 s substeps per hold.
python -m mujoco_reference sample --sensor-root "$SENSOR_ROOT" --output outputs/reference/samples --clock fixed

# Check an exported input/action/successor without the training archive.
python -m mujoco_reference validate outputs/reference/samples/F_transition.json
```

These audits read checkpoints trained before v3.8, which saw the requested hold. Their settings carry no `hold_disclosure` key, so the environment keeps the disclosed-hold interface for them.

`--episodes 5` supplies five paired evaluation episodes per cell. Keep the training seed separate from the evaluation-episode seed when later aggregating results. `--video-episodes 1` records the first episode of each cell using the existing renderer. `--max-seconds .5` is an installation diagnostic and is labeled `diagnostic_cutoff`; it cannot establish course success. Use a new `--output` directory to preserve earlier results.

For a larger planning budget, pass `--candidates 48 --iterations 3 --horizon-steps 40 --replan-seconds .04`. That increases evaluation compute and still runs no training. It also changes the baseline specification and should receive a separate output directory.

To match the uploaded pure-CPHT planner's search settings, use `--candidates 48 --elites 8 --iterations 3 --horizon-steps 20 --replan-seconds .04`. The checked eight-candidate comparison uses substantially less search, so its learned-MPC results are not a replay of the videos under their original planning budget.

The reader uses restricted Torch checkpoint loading with an explicit NumPy allowlist for the supplied legacy checkpoint format. The archive contains no CT-SAC `final.pt`, so a CT-SAC replay comparison cannot be produced from these uploads. CT-SAC runs trained by `run_experiments.sh` are evaluated in the main paired evaluation instead. Do not infer numerical evaluation results by digitizing its videos.

## Concrete pipeline input and received transition

The files under `validation/reference_samples/{fixed,irregular}/` contain newly executed terrain-state transitions. They are not reconstructed original video frames. Each is a local stress snapshot reset near x = 6 m, followed by .2 s of causal feedback warmup; it is separate from the full-course tests, which start at x = 0 m. Under irregular holds, warmup can end just after .2 s.

Each JSON separates:

- `pipeline_input`: timestamp; ordered raw observation and units; previous command; inferred scan; environment estimate, uncertainty, age and validity fields; requested hold; reference time; target speed.
- `decision`: normalized six-motor command, motor gears, physical motor torques, and solver diagnostics.
- `received_transition`: actual elapsed hold, executed physics-step count, successor q/v and raw observation, reward, and termination information.
- `predictions`: exact MuJoCo replay, learned pH prediction under the same held command, a separate audit-only learned prediction with true current terrain samples/root-location friction, and local mass/drift/contact-port diagnostics. Model weights remain fixed in both learned predictions.
- `substep_audit`: pre/post q/v, acceleration, mass matrix, bias, passive/actuator/applied/constraint forces, contact records, force-equation defect, and integration defects.
- `replay_audit_only`: full backend integration state, model snapshot, local terrain truth, version, and information-access label. These fields are excluded from ordinary policy inputs.

The 31 raw entries are q[0:9], v[9:18], aggregate constraint port[18:27], remaining COM distance/path length[27], remaining time/horizon[28], target speed/10[29], and the next requested hold/reference time[30]. The root-x coordinate is retained because terrain and the finite-course objective depend on position. The force port at a decision is measured with the previous held command. A newly selected command can change contact reaction immediately. The observation therefore does not supply the exact future contact port for every proposed action.

In `state` mode, the raw vector has 31 components and no external geometry sensor is used. The causal observer still supplies an inferred scan and friction estimate from original-state history. The two augmented modes have 60 components and need checkpoints trained with that feature contract. An augmented observation must not be appended to a 31-input SAC network at deployment.

## What constitutes a consistent state transition

First check finite values, declared shapes/feature order, actuator bounds, torque/gear mapping, and input/successor timestamps. Use the actual executed hold from the transition, which can be shorter than the requested hold after a terminal event. A successor's `next_h` refers to the next decision.

For each physics substep, the force equation is

\[
M(q_k)\ddot q_k+c(q_k,v_k)
=Ba_k+\tau_{{\rm applied},k}+\tau_{{\rm constraint},k}.
\]

The exported force audit uses the newly issued command. In this wrapper, the pre-step external-force hook runs before dm_control installs that command; reading its old `ctrl` without overriding it would audit the previous command.

For the installed Euler backend with implicit joint damping, the substep is

\[
v_{k+1}=v_k+\delta\,[M(q_k)+\delta D]^{-1}M(q_k)\ddot q_k,
\qquad q_{k+1}=q_k+\delta v_{k+1}.
\]

Here \(D\) is joint damping and `qacc` is the instantaneous forward acceleration, which already includes damping at the old velocity. If Euler damping is disabled, use \(D=0\). For this scalar-joint cheetah chart, the coordinate addition is valid. Quaternion configurations would require MuJoCo's position integration map. A naive `v_next = v + delta*qacc` check would misclassify this backend's damping treatment as a transition error.

Compose all executed .002 s substeps with the command held fixed. The portable replay restores `mjSTATE_INTEGRATION`, including warm start and applied inputs, and matches dm_control's split-step convention. Restoring only q/v does not reproduce the complete numerical state. Binary `.mjb` snapshots require the recorded MuJoCo version; this validator enforces that requirement.

The validator rejects corrupted successor observations, timing, action bounds, torque mapping, force equations, and replayed substeps. A successful exact replay validates data plumbing and the backend transition. It does not prove that a learned contact model predicts the same transition. The exported learned-model errors are a separate check.

The audit-only truth-environment prediction helps separate environment estimation from contact closure. It replaces the estimated scan heights/slopes and friction at the current root location while retaining the learned mechanics/contact parameters. It does not replace the entire contact constitutive law, geometry interpolation, or integrator with MuJoCo. A remaining prediction error can therefore arise from those components. A single root-location friction value also cannot represent simultaneous contacts on different materials. This diagnostic is not a sensor-matched controller result.

## Reading the outcome

The checked results and single-episode scope are recorded in `validation/mujoco_reference/report.md`, `results.json`, and `comparison.png`. The uploaded training-log audit is in `upload_audit.json`. No heavy training was launched.

If both learned and exact current-state feedback fail, investigate reference motion, underactuation, target weights, and gait phase before increasing physical-model capacity. If MuJoCo MPC improves progress while remaining incomplete, inspect planning horizon, search budget, and the restricted periodic reference. If mass/drift agree locally and contact force or held-state prediction differs, inspect contact closure, friction belief, force assimilation, and integration separately. A low projected LS residual verifies a local optimization solve; it does not imply feasible desired acceleration or stable locomotion.

The current F/J experiment samples a regime per episode. J alone does not establish adaptation to an abrupt within-episode change. The sensing pipeline's friction-event tests remain available for a later, separate perturbation experiment.

## Sources

- [MuJoCo computation: equations, constraints, and numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html).
- [MuJoCo API: forward dynamics, state access, copied data, and finite differences](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
- [dm_control engine source: split-step convention](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py). The checked installed version is 1.0.50.

This extension uses native MuJoCo and finite differences. It does not claim to reproduce SuperMimic training. The differentiable experiment and its source attribution are in [DIFFERENTIABLE_CPHT.md](DIFFERENTIABLE_CPHT.md).
