# A finite running course for the F/J benchmark

## Task and state

The new `path_speed` objective finishes a forward course while tracking a constant reference speed.
The cheetah has the original nine generalized coordinates, nine velocities, six geared motors, and command range `[-1,1]^6`.
Its motion plane has no lateral steering degree of freedom.
The course follows the forward horizontal axis through the selected terrain.
Distance and speed use the same whole-robot subtree centre of mass (COM) as the source cheetah speed sensor.

Let $x_c(t)$ be its horizontal COM position. The fixed course starts at $x_s$ and ends at $x_g=x_s+L$.
The declared target is $v_\star=g>0$, and $v(t)=\dot x_c(t)$.
These are horizontal distance and speed. Surface arc length can differ on a slope.
The default values are

$$
x_s=0\ {\rm m},\quad L=20\ {\rm m},\quad v_\star=3\ {\rm m/s},\quad T=12\ {\rm s}.
$$

The v3.2 path presets use the CT terrain heightfields described in `TERRAIN_ALIGNMENT.md`.
F keeps course seed 1 and one target. J draws a new course per episode and also uses 3 m/s by default.
This keeps environmental transfer separate from target-speed transfer.
The optional J `task` cohort tests 2.7 and 3.3 m/s.
The earlier `native_run` objective remains available for the source-style benchmark.

A pose is sampled and settled with the source initialization procedure.
The path reset aligns the COM with the requested course position, preserves torso clearance over the local terrain,
and clears generalized velocity. Policy training/evaluation starts at the course start.
Calibration can use declared intermediate course positions and initial forward speeds.
These are reset conditions and supply no extra policy actuator.

## Goal, fall, and deadline

Goal completion requires $x_c\ge x_g$ while the robot satisfies the task's upright criterion.
A fall occurs when torso clearance is below 0.20 m or absolute root pitch exceeds 1.5 rad.
A fall has priority when the goal and fall conditions occur at the same step.
The deadline ends an incomplete course.
These checks run at each physics substep; a goal can shorten an action hold.
The course ends at forward speed. It has no additional stop-at-goal requirement.

Goal/fall events are true terminations. Deadline truncation also has zero continuation because the objective is finite horizon.
The information state includes the remaining deadline.
The decision cap is large enough to avoid an additional earlier deadline under the configured minimum hold.
Task detection is quantized by the physics step, 0.002 s by default.

## Reward and units

The physical running cost is

$$
\ell(v,a)=4\left(\frac{v-v_\star}{v_\star}\right)^2
 +0.01\frac{\lVert a\rVert^2}{6}+0.05
 +0.1\max\left(-\frac{v}{v_\star},0\right).
$$

The command term is a control-effort proxy. It supplies no mechanical work or energy certificate.
The speed term penalizes both underspeed and overspeed.
The time term encourages progress, while the higher tracking weight keeps the speed target central.
The objective permits a trade-off among completion, tracking, and effort.
It imposes no hard constant-speed constraint.

Use $\beta=0.05\ {\rm s}^{-1}$ and the reference interval $t_{\rm ref}=0.01$ s:

$$
\gamma_{\rm ref}=e^{-\beta t_{\rm ref}},\qquad \gamma_h=e^{-\beta h}.
$$

For an action stopped after actual elapsed duration $h$, the measured interval return is

$$
R_h=-\int_0^h e^{-\beta s}\ell(v_{t+s},a)\,ds+e^{-\beta h}B_{t+h}.
$$

The goal event pays $B=5$. Timeout pays $B=-2$.
An early fall also pays the remaining stationary failure cost:

$$
B_{\rm fall}(t)=-2-(4+0.05)\frac{1-e^{-\beta(T-t)}}{\beta}.
$$

This is an absorbing failure tail. It prevents an early fall from avoiding the remaining course cost.
The episode ends immediately; the code adds the tail as one event instead of simulating a stationary failed robot.
The event is paid once.

The source wrapper represents endpoint reward per reference interval, so its smooth rate channel is $-t_{\rm ref}\ell$.
Its interval integration then returns the physical-time cost integral above.
The measured event is added once at the terminal endpoint with its interval discount.
The rate-proxy channel includes the duration-normalized event for source-style targets.
For every path-task CT adapter, a measured terminal transition explicitly uses $Y=R_h$.
This finite boundary override prevents division of a terminal value by a shortened hold.
The override and its correction are logged. The original `native_run` targets retain their earlier equations.
The new terminal-event task is a declared extension of the source benchmark.

The path-task entropy coefficient is `alpha_reference=0.002`, corresponding to 0.2 per physical second.
The start-of-interval entropy convention uses the commanded holding time.
Clocks remain exogenous and shared by all methods.

## Measurements and the two holding times

Every actor receives the same robot/terrain measurements as in its selected F/J subcase.
A virtual COM odometry sensor adds

$$
o_k^{\rm goal}=\frac{x_g-(x_c(t_k)+\epsilon_k)}{L},\qquad
\epsilon_k\sim\mathcal N(0,(0.01\ {\rm m})^2).
$$

It supplies one synchronous, zero-latency channel with zero acquisition fee.
Its samples are counted. It gives no material-map ID or parameter label.
It is a declared synthetic simulator sensor and requires its own calibration for a physical robot.
The default contact observation grows from 31 to 32 components.
Position-only F grows from eight to nine. A shared causal encoder is fitted for the path task.

Replay retains both:

- `h_requested`: the command duration available to the actor before action selection.
- `h`: the elapsed duration actually observed, which can end early at a goal or fall.

The current information vector and actor entropy use `h_requested`.
History retains actual elapsed time. Measured rewards and transition discounts use actual `h`.
This keeps a terminal interval from replacing the information that was available before the action.

## Shared preparation and terminal coverage

Preparation uses the same measured records and frozen information map for every compared method in a case/seed.
Ordinary calibration reset positions cover fractions $0,0.2,0.4,0.6,0.8,0.975$ of the course.
Their initial COM speed is sampled from $0,0.5v_\star,v_\star$.
The source pose is settled first; the declared reset then places it and sets its velocity.
The fixed goal remains $x_g$ in every calibration episode.

Before ordinary acquisition, bounded goal probes seek four measured successful
episodes per known target speed. Each starts with speed $v_\star$ at
$x_g-\min(0.01L,0.25v_\star\Delta t_{\rm physics})$.
The controller supplies the same bounded random-action law. MuJoCo supplies the
actual outcome, reward, and elapsed time. Each task has at most 32 attempts and
each attempt has at most four decisions. These decisions count inside the
declared calibration budget.

Whole episodes are stratified by known task and observed terminal outcome.
Both partitions must contain at least one measured goal episode for each task before observer/model fitting proceeds.
Preparation stops with an actionable coverage error when this check fails.
It saves `acquired.pt` and `calibration_acquisition.json` before fitting, so a
later fitting retry can reuse measured records under the same contract.
`README_CALIBRATION_FIX.md` defines the protocol and its configuration keys.
The exposure audit records task outcomes and reset conditions.
This coverage check supplies no global exploration or posterior-sufficiency theorem.

The matched CT presets use full difficulty in calibration, policy training, and evaluation.
They retain the source's stock robot and contact response.
The earlier mesh backend retains its declared J difficulty curriculum when selected.
Evaluation uses the course-start reset for both physical backends.

## Learned live-transition operator

A finite course needs an action-dependent termination model.
For each learned member, predict a conditional live transition row $P_m(I,a)$,
expected interval reward $r_m(I,a)$, and survival probability $s_m(I,a)\in[0,1]$.
Fit survival by binary cross entropy from observed terminations/deadlines.
Fit conditional next-node rows on live records, reward on every record, and terminal-value probes with survival included.

For an ensemble with a common current action, average the joint live kernels:

$$
\Pi(I,a)=\frac1M\sum_{m=1}^M s_m(I,a)P_m(I,a),\qquad
\bar s=\frac1M\sum_m s_m,\qquad \bar P=\Pi/\bar s.
$$

When survival is numerically zero, the conditional row is immaterial to continuation.
The code retains a normalized fallback row and zero survival.
The frozen twin-critic backup uses

$$
(\mathcal Uq)_{jik}=\bar r_{ik}
 +\gamma_{h_{ik}^{\rm requested}}\bar s_{ik}
 \sum_\ell\bar P_{ik\ell}\frac1K\sum_u
 \left[\min(q_{1\ell u},q_{2\ell u})-\alpha_{h_{\ell}^{\rm requested}}\log\pi(a_{\ell u}|I_\ell)\right].
$$

The known deadline also sets continuation to zero.
Positive rows and survival preserve the finite operator's sup-norm contraction for a fixed model and policy.
Euler, the pH residual update, and the direct finite solve share this exact operator.
Their existing free twin-critic recovery and numerical admission rules remain in force.
The learned survival/reward/transition errors and approximate history closure remain empirical modelling errors.
A contraction of the fitted operator supplies no physical policy-optimality or safety certificate.

## Train and deploy algorithm

```text
Inputs: case, environment/sensor family, fixed course, target speed, deadline,
        selected method subset, training seeds, interaction and fitting budgets.
Outputs: checkpoints, per-episode training records, paired task evaluations,
         physical-time videos, plots, and training-seed aggregates.

1. Validate the requested methods and task settings. Generate only those jobs.
   Prepare one shared artifact per case/seed.
2. Collect bounded measured goal probes, then ordinary random-action records,
   inside the calibration budget. Retain commanded and elapsed holds. Split
   whole episodes by task/outcome and verify per-task goal coverage in both parts.
3. Fit the shared causal encoder. Fit reward, conditional live rows, survival,
   and probe backups. Freeze the information map and node bank.
4. Collect policy episodes from the course start. Step contact dynamics at each
   physics substep. Emit goal/fall/deadline events once and record the true stop.
5. For each critic block, freeze model, actor, entropy, nodes, and quadrature.
   Apply the chosen measured/source target or common finite operator solver.
   Use the exact measured terminal return for a path CT boundary transition.
   Record numerical gates, distillation debt, solver work, and rejected work.
6. Update the actor/targets on the method's schedule. Save recovery state at
   episode boundaries. Log completion and physical-time speed metrics.
7. Evaluate candidate checkpoints on validation seeds and all selected clocks.
   Rank completion rate first and relative speed RMSE second. Save the selected
   checkpoint identity without changing its weights.
8. Freeze learning. Evaluate on paired test seeds and the declared J holdouts.
   Capture MP4 frames within the same physics rollout. Save task JSON and PNGs.
9. Aggregate known task strata first, then independent training seeds.
   Report completion, speed error, speed-band fraction, terminal causes,
   conditional completion times, and computation costs.
```

The default operator remains frozen during policy comparison.
The optional online operator uses permitted replay and held-out calibration admission.
Its survival term and live-row masking are included in that admission score.
Rejected candidate updates still count as work.
The pipeline introduces no learned sensing, learned timing, or physical force-measurement privilege.

## Metrics and interpretation

The primary metric is completion probability before the deadline while upright.
For each actual episode duration $\tau$, compute

$$
{\rm RMSE}_v=\sqrt{\frac1\tau\int_0^\tau(v-v_\star)^2dt},\qquad
f_{\rm band}=\frac1\tau\int_0^\tau\mathbf1\{|v-v_\star|\le0.1v_\star\}\,dt.
$$

The physics-step quadrature weights time directly, so unequal holding times do not change the metric weights.
RMSE includes startup acceleration.
Report failed episodes alongside successful ones. A short failed episode can have small speed error and zero completion.
Successful-episode speed statistics and completion times are also reported separately.
Their values remain undefined when the run has no successes.
Arrival-time error is measured against initial remaining distance divided by target speed.

JSON retains the older normalized `time_score` for compatibility.
Path aggregation ranks `success_rate` and reports speed metrics separately.
It also retains discounted task return, path traces, contact slip, penetration, and cost ledgers.
The one-second terminal video card is a display hold, excluded from physical time and all task metrics.

A new objective/observation/terminal model receives schema 4 and a new run identity.
The saved actor/observer from an older task can still be replayed under that saved task.
Those weights supply no newly trained course-completion policy.
