# Shared sensing and environment assimilation: v3.5

This release adds an optional experiment to compare CT-SAC and Pure CPHT with three information contracts. Earlier runners, cases, checkpoint formats, and default observations remain available. Run the new experiment with `scripts/run_sensing.sh`. Omit the `sensing` options and `--information` argument to use an earlier setup.

Pure CPHT remains the experimental mechanical pH feedback/MPC method described in `PURE_CPHT.md`. It learns positive mechanical parameters and a contact approximation, then computes actuation through feedback and motion search. Its controller has no SAC actor or critic. The manuscript's HJB/2PBVP value solver remains a separate implementation.

## 1. Questions and information modes

The robot retains nine configuration variables and six motor commands. The new observation features describe the environment; they add no robot joints or canonical momenta.

The **original state** in this experiment is the 31-channel physical-first observation:

\[
o_k=[q_k\in\mathbb R^9,\ v_k\in\mathbb R^9,\
\tau_{\mathrm{constraint},k}\in\mathbb R^9,\
d_{\mathrm{goal}},t_{\mathrm{left}},v_*,h_{\mathrm{requested}}].
\]

These state and aggregate-force measurements are instrumented simulator measurements. The original CT-RL paper uses a different observation interface. This experiment tests adaptation under a common instrumented interface.

Choose one mode:

1. `state`: CT-SAC receives the original 31 channels. CPHT receives the same original measurements and infers a contact surface from their causal history and public robot geometry. Additional terrain probes and material sensors are disabled. Its predicted contact points come from known capsule geometry and state-dependent Jacobians. Initial terrain and friction priors are explicit.
2. `state_belief`: both methods receive the original channels plus 29 features from the same state-history estimator. It has no external terrain sensor. This mode tests whether explicit assimilation and memory help CT-SAC. CPHT's inferred environment and actions match `state` for matched seeds and data; the extra vector exposes its already available environment estimate to CT-SAC.
3. `sensed_belief`: both methods receive the same 60-channel schema as `state_belief`. The shared estimator also receives noisy height/slope rays. Their sampling period, delay, and dropout are declared in the options. Direct friction or material labels remain disabled.

Compare `state` with `state_belief` to assess explicit state-history processing. Compare `state_belief` with `sensed_belief` to assess added geometric sensing. Compare the two methods within each mode to assess the controller/model choices under the same permitted information.

The earlier `cases/pure_cpht.json` retains its 45-channel raw terrain-probe interface. Earlier physical-first runs retain their original 31-channel interface. A checkpoint keeps its trained observation schema; new modes require separate training.

When `sensing` is present, its mode owns the observation contract. The older `pure.terrain_probes` flag applies to the legacy setup and is ignored by this new layer.

## 2. Estimator and observation features

The estimator maintains a local quadratic surface around the current torso x position:

\[
h(x_0+\xi)=a_0+a_1\xi+\tfrac12a_2\xi^2,
\qquad \kappa(x_0)=\frac{a_2}{(1+a_1^2)^{3/2}}.
\]

A Kalman update combines geometric measurements. As the robot moves, a change of polynomial origin transports the coefficients and covariance. In state-only modes, loaded contact candidates provide sparse height evidence. Slope and curvature remain uncertain when these measurements provide little spatial support. In sensed mode, the latest acquired height/slope samples also support a piecewise local scan. Large quadratic-fit residuals clear the curvature-valid flag, which helps represent a step as an edge.

The 29 appended features have a fixed order in `environment_belief/config.py`:

- seven relative terrain heights and seven terrain slopes at the declared offsets;
- signed local curvature and a positive friction estimate;
- height, slope, curvature, and friction standard-deviation estimates;
- geometry and friction evidence ages;
- geometry-valid and friction-valid flags;
- a loaded-sliding evidence flag and normalized normal load;
- external-geometry-frame age, external-frame validity, and curvature validity.

The physical sensor samples on a clock independent of control decisions. Sampling occurs at the next available physics step; timing error is bounded by one physics step. Delayed frames retain their acquisition timestamp and world coordinates. The observer consumes a frame only after its delivery time. The geometry noise generator and hidden-intervention generator are separate. Changing the information mode therefore preserves the physical course and intervention draw for a paired episode.

The controller receives no map object, material ID, true friction coefficient, future frame, switch timestamp, or audit record. The sensor boundary queries simulator geometry to emulate the declared rays. The observer itself has no simulator or terrain-map interface. Existing task scoring and audit code can still access truth.

### Friction inference

Public articulated kinematics give candidate contact points and point Jacobians. The observer selects a small low-point contact set and fits point forces to the measured aggregate constraint-force port by regularized least squares. It omits joint-force channels near joint stops. This inverse is independent of the current friction estimate.

For a candidate contact, the filter checks normal load, reconstruction residual, tangential speed, a contact dwell time, and dissipative force direction. A sufficiently informative sliding segment supplies the approximate measurement

\[
\log\mu \simeq \log\frac{|F_t|}{F_n}.
\]

The filter applies a bounded update in log-friction space. During flight, weak contact, sticking, large inversion residuals, or impact transients, it retains uncertainty. A new spatial cell restores broad friction uncertainty. Sticking implies a force-cone inequality and supplies no unique coefficient; this first implementation skips a sticking likelihood update.

This is an approximate estimator. Ambiguous contact allocations, soft-contact regularization, geometry errors, and simultaneous contacts with different friction can bias it. Its covariance is a diagnostic estimate. Statistical coverage and parameter identifiability require separate validation. A single local friction estimate represents the current contact neighborhood; this release has no per-foot material map or RGB material classifier. Contact stiffness and damping remain learned global closure parameters.

## 3. How CPHT consumes the same estimate

CPHT uses the inferred scan to determine gaps, active contact candidates, normals, and tangents. Its point Jacobian changes with the articulated configuration and the contact lever arm. For world point Jacobian \(J_{x,i}\),

\[
\dot p=-\partial_qH_\theta-D_\theta\partial_pH_\theta+Bu+
\sum_i J_{x,i}(q,r_i)^\top(n_i\lambda_{n,i}+t_i\lambda_{t,i}),
\qquad |\lambda_{t,i}|\leq\widehat\mu_k\lambda_{n,i}.
\]

The current belief supplies \(\widehat\mu_k\). Changing this environment input preserves the calibrated robot parameters. The controller also retains the existing measured-port residual correction. Its predictive rollouts use its learned mechanical/contact model and the inferred scan.

CT-SAC uses the belief features as actor/critic inputs in the two belief modes. Its replay stores the causal current/successor vectors collected during interaction. Normalization uses calibration-training episodes. Evaluation freezes actor/critic, mechanics, contact-calibration, and estimator parameters. The belief state updates online. CPHT may also recompute motion proposals through MPC; `FEEDBACK_ONLY=1` freezes its saved periodic motion parameters and retains feedback.

## 4. Training supervision and fairness

The new presets set `pure.contact_supervision=aggregate_ports`. CPHT fits its positive contact closure against the measured aggregate constraint-force port using its own causal scans. This fit uses no simulator contact point, normal, penetration, point force, or friction label. Public robot geometry supplies its candidate points and Jacobians. Earlier CPHT options retain point-label fitting.

Both methods share one calibration artifact within each mode and seed. Calibration contains exact q/v, acceleration, actuator/aggregate-force labels, and measured transitions. CPHT uses the physics labels to fit mechanics; CT-SAC uses the shared transition replay. Thus, this is a common deployed-information comparison with instrumented mechanics supervision available in the shared calibration dataset. It does not establish learning from ordinary joint encoders alone. A future sensor-realistic experiment should estimate acceleration and force ports from the same noisy proprioceptive history for both methods.

For the original-state comparison:

1. Train **both** methods with `INFORMATION_MODES=state`, the same task, seed set, calibration budget, course generator, intervention distribution, action limits, and interaction budget.
2. Keep CPHT's public geometry/model prior explicit. Its inferred scan uses permitted state history. It receives no private terrain probes during preparation, training, or deployment.
3. Evaluate frozen checkpoints on the same episode seeds, courses, speeds, clock, sensor contract, and hidden interventions. Compare both on fixed and irregular clocks separately.
4. Report completion, speed RMSE, falls, task return, actual physics steps, training updates, and planning work. Equal interaction budgets permit different computation costs; report those costs separately.
5. Use `state_belief` as the memory/assimilation control. The `state` CT-SAC actor is feedforward, while CPHT carries observer/controller state. A difference in that mode can reflect memory, physical priors, supervision use, and control design. The paired belief experiment helps separate these effects. A recurrent CT-SAC baseline remains a useful future control.

Match physical interaction time as well as decision count when making strict sample-efficiency claims. The fixed training clock makes decision budgets comparable; early termination and final partial intervals can change actual physics-step totals. Those counters are saved. Use independent training seeds as statistical replicates; evaluation episodes within one trained seed are paired test samples.

## 5. Dynamic courses and hidden changes

F retains its fixed base course; J draws fresh base courses as before. Spatial height and friction patches change the contact conditions as the robot moves. The new options can also change friction during an episode:

```json
"friction_events": [
  {"time": 4.0, "scale": 0.45},
  {"time": 8.0, "scale": 1.0}
],
"friction_event_jitter": 0.6,
"friction_scale_jitter": 0.15
```

Each scale is absolute relative to the sampled course. Time jitter is uniform in seconds; scale jitter is Gaussian in log scale. Draws are independent of sensor availability. Randomized timing reduces the value of memorizing a switch schedule from the clock. With these events enabled, F means a fixed base course with additional declared dynamic interventions. Set an empty event list for the earlier fixed physical regime.

The backend updates ground-contact friction before a physics substep. The change is applied at the next physics boundary and recorded only in audit output. Height geometry is static within each episode in this release. Changing elevation while the robot moves would require an additional moving/deforming-ground model.

## 6. Run commands

Use the existing environment in `requirements-physical-first.txt`. It leaves the installed Torch/CUDA build in place. Run commands from the extracted project folder. The new script is dry by default.

Inspect the complete plan:

```bash
CASE=both STAGE=prepare DRY_RUN=1 bash scripts/run_sensing.sh
```

A short sensor-only precheck requires no checkpoint or training:

```bash
MUJOCO_GL=egl python preview_sensing.py \
  --case J --information sensed_belief \
  --options cases/sensing_smoke.json \
  --config cases/physical_first_smoke_task.json \
  --seconds 0.06 --video --out prechecks/sensing
```

It writes a headless MP4, an estimate/reference PNG, and JSON with frame ages, counts, and interventions. Actions are zero by default. Use `--random-actions` to exercise the sensor with random actuation. This precheck demonstrates interfaces and rendering.

Prepare and train the matched experiment on your server:

```bash
export CASE=both
export SEEDS=0,1,2
export METHODS=ct_sac,pure_cpht
export INFORMATION_MODES=state,state_belief,sensed_belief
export OPTIONS=cases/sensing.json
export MUJOCO_GL=egl

STAGE=prepare DRY_RUN=0 bash scripts/run_sensing.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_sensing.sh
```

Choose `DEVICE=cuda` if supported by your environment. Select only `METHODS=ct_sac` or `METHODS=pure_cpht` to reduce cost. For the requested original-state-only comparison, set `INFORMATION_MODES=state` for both stages. The runner saves separate mode/case/method/seed directories. Original run directories remain separate.

Evaluate checkpoints and record videos:

```bash
STAGE=eval DRY_RUN=0 DEVICE=cpu EPISODES=5 VIDEOS=1 \
  CLOCKS=fixed,irregular bash scripts/run_sensing.sh
```

To evaluate an explicit checkpoint:

```bash
python eval_physical_fj.py --checkpoint-file /absolute/path/to/final.pt \
  --clocks fixed,irregular --episodes 5 --video-episodes 1 --gl egl
```

Evaluation writes task JSON/PNG, an assimilation PNG for the first episode, causal estimate/audit traces, and the requested videos. Force/geometry truth remains in these diagnostic records. The local root-patch friction reference can differ from the contact coefficients at the front and rear feet; active contact-pair truth is available in `contact_audit`.

The traces also contain active contact coefficients and slip speeds. Assimilation summaries report a normal-load-weighted friction error on loaded sliding contacts. Curvature uses a slope finite-difference reference over +/-0.05 m, with large slope jumps excluded from its summary. This reference depends on the terrain discretization. Missing evidence has a validity flag; diagnostic age plots omit missing values.

Test a different hidden change schedule and sensor degradation while retaining weights and observation dimensions:

```bash
STAGE=eval DRY_RUN=0 SENSING_TEST=cases/sensing_test_shift.json \
  EPISODES=5 VIDEOS=1 bash scripts/run_sensing.sh
```

The same overrides must be used for compared methods. `CASE_TEST=J` and `TERRAIN_TEST=steps` retain their earlier meanings. Evaluation rejects a request to switch a checkpoint between information modes. Sensor-test outputs include a settings hash to avoid overwriting the default evaluation.

Compare one case/cohort/clock/test condition at a time:

```bash
python compare_sensing_fj.py --evaluations \
  runs/sensing/state/F/terrain/*/seed*/evaluations/final_FtoF_terrain_iid_fixed_n5.json \
  runs/sensing/state_belief/F/terrain/*/seed*/evaluations/final_FtoF_terrain_iid_fixed_n5.json \
  runs/sensing/sensed_belief/F/terrain/*/seed*/evaluations/final_FtoF_terrain_iid_fixed_n5.json \
  --out reports/sensing_F_fixed.json
```

The comparison checks course/task pairing, information contracts, calibration identity within each mode, sensor/intervention settings, training-seed sets, and decision budgets. It reports method differences within modes and information-mode differences within methods. The standard error uses independent training seeds.

## 7. Algorithm sketch

```text
Inputs: original-state contract, information mode, shared calibration budget,
        public robot geometry, sensor settings, train/test course distributions

Preparation:
  acquire shared causal transitions and instrumented mechanics labels
  fit normalization on training episodes
  CPHT: fit positive mechanics and aggregate-port contact closure

For each training episode:
  initialize shared estimator priors and the permitted sensor stream
  for each physics step:
    apply hidden environment changes in the backend
    acquire geometric rays on their own clock when enabled
    queue frames with their acquisition/delivery times
  for each control decision:
    transport the local environment estimate to the current pose
    assimilate delivered frames and current original-state measurements
    construct the versioned observation and inferred contact scan
    CT-SAC: sample action, store causal transition, update actor/critics
    CPHT: compute learned-model feedback/MPC action, collect mechanics data,
          refit the physical model at the declared training interval

Evaluation:
  freeze learned parameters and normalization
  update only causal belief/controller states and permitted MPC decisions
  record task outcomes and separate audit truth
  compare paired courses using independent training seeds
```

## 8. Verification and practical limits

`tests/test_environment_belief.py` checks a known curved surface, coordinate transport, contact/flight friction gates, independent MuJoCo Jacobians, state-only isolation, sensor timing, hidden switches, and short checkpoint training/evaluation for both methods in all modes. `VALIDATION_v3_5.md` records the executed checks.

The smoke runs contain four policy decisions per method/mode. They establish executable interfaces and finite numerical operations. Full-course performance, robust friction identification, calibrated uncertainty coverage, and an advantage from the pH model require the server experiments above. No expensive training is included in this revision.
