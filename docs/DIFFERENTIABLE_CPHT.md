# The pH engine: equations of the differentiable CPHT experiment

## What is adapted

The four methods isolate motion search and motion-selector learning. The inspected source is the independent SuperMimic reproduction pinned in `third_party/supermimic_reproduced/SOURCE.json`.

| Source pattern | Cheetah adaptation |
| --- | --- |
| Imagined policy trajectories | Held-action trajectories through calibrated pH mechanics/contact |
| Frozen model during policy backward pass | Fixed model parameters with active state/action Jacobians |
| Analytic policy gradient | Gradient shooting or a selector for 19 motion coefficients |
| Survival-weighted return | Smooth fall/goal approximation and absorbing failure tail |
| Terminal critic | Task-value residual around a stationary-cost prior |
| Detached TD(lambda) fitting | Variable-duration physical-time discounts |
| Logged simulator starts | Training-calibration and current on-policy causal information |

The model is fitted by positive mechanical/contact calibration and shared across methods. This experiment does not fit a generic neural world model on eight-step windows, refit the model after every policy update, use a human-motion dataset, or reproduce the humanoid experiments.

## Layer I: task and information

The configuration is `SE(2) x SO(2)^6`. It has nine configuration variables and nine conjugate momenta:

$$
X=(q,p,\mathcal E,m),\qquad p=M_\vartheta(q,m)v,\qquad a\in[-1,1]^6.
$$

Environmental variables receive no additional mechanical momenta. Six normalized motor commands act through the known gears 120, 90, 60, 90, 60, 30; the three root coordinates remain unactuated.

The observation is the existing instrumented 31-channel state/force/task interface or its 60-channel belief extension. This differs from source CT-RL's 17-channel observation. Calibration additionally uses simulator acceleration/force labels; count that supervision.

The modes are `state`, `state_belief`, and `sensed_belief`. The latter adds acquired geometric evidence; the other two share the original evidence with different exposure of the causal observer state. Friction estimates and covariance outputs remain approximate.

The path objective retains the nominal 20 m goal, 3 m/s speed, 12 s deadline, and original hard fall/goal payments. F fixes the base course; J samples new courses. Hidden friction interventions are an additional declared test condition.

## Layer II: model and bounded feedback

The predictor retains

$$
H_\vartheta(q,p)=\tfrac12p^\top M_\vartheta(q)^{-1}p+U_\vartheta(q),
$$

$$
\dot q=\partial_pH_\vartheta,\qquad
\dot p=-\partial_qH_\vartheta-D_\vartheta\partial_pH_\vartheta+Ba+J_c^\top f_c.
$$

The 32 mechanical coefficients remain positive. Contact uses the existing changing candidate geometry, regularized friction-cone QP approximation, and causal residual correction. A new optional batched friction argument accepts each replay start's inferred material context.

There are six reference offsets, six sine amplitudes, six cosine amplitudes, and one frequency:

$$
q_\psi^{\rm ref}(\varphi)=c_\psi+A_\psi\sin\varphi+B_\psi\cos\varphi,\qquad
\dot\varphi=2\pi\nu_\psi.
$$

The common acceleration-feedback decoder solves

$$
a^*=\arg\min_{a\in[-1,1]^6}
\tfrac12\|W^{1/2}(M_\vartheta^{-1}Ba+a_0-a^\star)\|^2
+\tfrac{\lambda}{2}\|a\|^2.
$$

Positive ridge gives a unique fixed-state quadratic optimum. Finite active-set iterations retain their residual.

Shooting differentiates return with respect to the 19 reference parameters, clips gradients, projects parameters to their bounds, and retains the best finite predicted candidate. Invalid gradients preserve the previous reference and produce a diagnostic.

The selector instead computes

$$
\psi_k=\mathcal N_\theta(I_k,\sin\varphi_k,\cos\varphi_k).
$$

Its last layer starts at zero around the same seeded periodic reference. Selector and search methods retain the same reference limits and feedback decoder. References refresh on the declared replanning clock.

This is a structured policy family. It supplies no exact HJB-optimality, Hamiltonian-shaping, passivity, or stable-limit-cycle certificate.

## Layer III: physical time and environmental approximation

Actual physics uses 2 ms MuJoCo steps. The ordinary control interval is 10 ms. Commands remain fixed across predictive substeps.

Holds are unannounced (paper Sec. 2.6). With the default `hold_disclosure: false`, every policy input and the imagined rollout use the nominal hold $h_{\rm nominal}=0.01$ s, and the next hold is drawn only after the command is chosen. The realized hold is recorded after each step and used for discounting and reward accounting. For `L` prediction intervals the imagined duration is $L\,h_{\rm nominal}$. Irregular-clock evaluation measures the cost of this nominal-hold approximation.

Setting `hold_disclosure: true` (or `--disclose-holds`, `DISCLOSE_HOLDS=1`) restores the v3.7 interface, where the current requested hold enters the policy input and the first imagined interval, giving $h_{\rm current}^{\rm req}+(L-1)h_{\rm nominal}$. Use it only as an ablation.

Imagined trajectories retain the current world-registered scan and friction estimate. Predicted pose transports height/slope features; evidence ages increase. Predicted pose/velocity and force ports replace their measured counterparts. No future measurement, private map, intervention timestamp, or true friction enters the selector.

This is a fixed-local-belief point approximation, rather than a differentiable joint Bayesian filter. Deployment continues to use the real causal observer and delivered measurements.

The contact solver now reports a dimensionless residual using its actual projected iteration:

$$
r_{\rm rel}=
\frac{\|f-\Pi_{\mathcal C}(f+(\mathrm{rhs}-W_cf)/L_c)\|_\infty}
{\max(1\ {\rm N},\|f\|_\infty)}.
$$

The new comparison uses this residual. Legacy residual fields remain unchanged. Feedback-box residuals use normalized command coordinates.

Velocity error is a coordinate-scaled L2 diagnostic, with scales 1 m/s for translations and 1 rad/s for rotations. Translation and angular-position errors are also recorded separately. Diagnostic thresholds are configurable and provide no convergence certificate.

## Layer IV: gradient learning and continuation

For interval payment `R_j`, physical time `t_j`, and soft survival `s_j`,

$$
\widehat J(\theta)=
\mathbb E\left[\sum_{j=0}^{L-1}e^{-\beta t_j}s_jR_j+
e^{-\beta t_L}s_LV_\nu(I_L)\right].
$$

The terminal term appears only in `selector_value`. Differentiable event probabilities are analytic approximations, with the absorbing stationary-cost tail retained for falls. Simulator evaluation keeps the actual hard events.

The critic represents task return and has a separate role from mechanical storage. Its weights are frozen during selector backpropagation while derivatives through its inputs remain active. Then detached imagined trajectories supply

$$
Y_j=R_j+\gamma_j c_j[(1-\lambda)V_{j+1}+\lambda Y_{j+1}],
\qquad\gamma_j=e^{-\beta h_j}.
$$

Model, normalization, observer parameters, selector, and critic weights remain frozen at evaluation. The critic is used during training, without a value query to choose deployed commands.

The tensor rollout is `cpht_compare/rollout.py`. Top-k contact selection, cone faces, active input bounds, clipping, and event approximations make it piecewise smooth. A directional finite-difference check compares

$$
d^\top\nabla_\psi\widehat J
\quad\text{with}\quad
\frac{\widehat J(\psi+\epsilon d)-\widehat J(\psi-\epsilon d)}{2\epsilon}.
$$

Agreement checks the implemented surrogate derivative on the tested branch. It does not validate true simulator gradients or improvement on the physical plant. Finite contact iterations affect both prediction and derivative.

## Failure-mode suite

| Condition | Physical/information change | Target |
| --- | --- | --- |
| `nominal` | Existing test distribution | Baseline |
| `layout` | Held-out course/layout stream | Geometry transfer |
| `friction_change` | Hidden friction switches | Assimilation/adaptation |
| `sensor_shift` | More delay/dropout | Sensing dependence |
| `steps` | Step terrain | Contact switches/extrapolation |

The suite supports F/F, J/J, and F-to-J transfer on fixed and irregular clocks. Sensor-shift tests are omitted without external sensing. Smoke changes occur at 0.02/0.06 s; ordinary tests use jittered 3/7 s changes. Intervention draws remain independent of sensor availability.

Reports separate actual task failures from administrative censoring, numerical failures, and inference limits. They flag actuator saturation, weak progress, invalid gradients/trajectories, contact residuals, prediction error, scan extrapolation, and lack of informative friction evidence.

A predicted search gain alongside large one-step model mismatch is a warning about possible model exploitation. It is not a causal counterfactual comparison.

Completion rates omit administratively censored episodes. Missing runs and incomplete training-seed groups mark the protocol incomplete. Each independent training seed is a statistical replicate; episodes are averaged within seed.

Noisy physical-label learning, full posterior scenarios, dynamic terrain geometry, arbitrary robot topology, and a generic neural transition-model baseline remain separate extensions.

## Reproducibility and costs

One calibration dataset/model is shared across methods and horizons for each mode/case/seed. Equal decision budgets can produce different physical seconds after early termination; both are logged. All controller-specific rollout, search, and neural training work is additional computation.

Runs record observation, task, calibration/model hashes, seed, prediction horizon, and update counters. Aggregation checks course/task pairing and model/calibration/sensing identity.

Recovery stores learned weights, optimizers, replay starts, random generators, and counters at episode boundaries. SIGINT/SIGTERM/SIGUSR1 request a boundary save. The final checkpoint requires the declared interaction budget.

## Sources

- [SuperMimic paper](https://arxiv.org/abs/2610.28480), Sections 3.1-3.3 and Appendices B-D.
- [Authors' project](https://supermimic.github.io/).
- [Independent reproduction, pinned tree](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2).
- Inspected upstream files: `imagination/objective.py`, `imagination/policy_trainer.py`, `imagination/rollout.py`, `world_model/wm_trainer.py`.
- How to run every stage and scale: [EXPERIMENTS.md](EXPERIMENTS.md).

