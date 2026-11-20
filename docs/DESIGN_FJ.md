# F/J implementation and equation map

The source documents are identified by exact hashes in `cases/sources.json`.
The F/J labels come from the measurements note, Section 7, Algorithms 3–4.
The uploaded information-operator PDF supplies the SAC specialization in Sections 4.5–4.6 and the additional qualifications in Appendix C.

## State and data boundaries

Case F fixes robot, task, environmental law, static parameters, and sensor law.
Random starts, commands, and received observations remain variable.
Case J draws friction and payload once per episode and keeps that draw fixed.
Its wind variant adds a dynamic environmental state.
The known speed target enters every learned map; hidden parameters remain in the simulator.

The main files have these responsibilities:

- `fjbench/environment.py`: measurements note (88)–(93), including calibrated gears, position-only noise, and the off-centre wind load.
- `fjbench/data.py`: the measured source record in Section 7.1 and causal histories in (65)–(66).
- `fjbench/models.py`: a common finite-history information approximation and positive action-conditional operator model, the Op branch of (75).
- `fjbench/calibration.py`: common observer/model fitting, separate calibration episodes, terminal-value probes, and predictive checks.
- `fjbench/agent.py`: source SAC/TD3 targets (24), (26), actor schedules, and the interval specialization in manuscript (39)–(40).
- `fjbench/finite.py`: matched direct iteration (83), the numerical phase in (86), and exact twin-critic residual recovery.
- `fjbench/runner.py`: Algorithms 3–4, costs, model-adoption checks, reset rules, and episode-boundary recovery.

The configuration retains the actual hold and the declared next hold.
The exogenous clock is sampled before the corresponding action and is visible to every method.
For a continuing objective, the actor uses causal information, known task, and queued hold.
For a time-only finite horizon, it also receives remaining time.
The latter configuration requires a decision cap large enough to leave the physical horizon as the binding deadline.

Raw replay stores observations and received rewards rather than cached GRU embeddings.
A current history contains no successor observation or current-interval reward.
The successor history appends those quantities after reception.
Validation episodes cannot enter later actor, critic, or operator fitting.
Ring-buffer overwrites preserve those checks.
Task-stratified index pools provide constant-time record insertion and sampling without scanning the full replay each decision.

The causal GRU does not represent a normalized posterior over physical parameters.
Its predictive consistency loss is a measured approximation diagnostic.
It supplies no posterior-error or closure certificate for (65)–(66).
The native F interface uses measured velocities; the position-only interface never uses simulator velocities as learning targets.
Reward observations can reveal motion and remain part of the permitted history.

## Learned operator

The common calibration has a fixed information-node bank, split by the known task descriptor.
Each model predicts reward and a softmax distribution over successor information nodes with the same task.
F uses one member. J averages three conditional model predictions.
All members receive the same current information and action.
They receive no hidden regime label.

The loss contains three terms:

1. Squared error in received source reward and measured interval return.
2. Cross-entropy against positive interpolation of the observed next information.
3. Squared backup error on bounded terminal-value probes.

The target interpolation itself is an approximation.
Probe reuse changes the tested function argument, while retaining the original number of physical records.
Held-out prediction and probe losses are empirical scores.
They do not provide the uniform total-variation error needed by manuscript (C.16).

The default frozen-model study shares this entire artifact across methods.
The online model study checks each proposed update on separate calibration episodes.
It compares the same held-out batch before and after the proposal, restores rejected model/optimizer states, and charges every attempt.
This is an empirical adoption rule, rather than a physical identification certificate.
The observer and node bank remain fixed in both studies.

## The finite twin-critic operator

Let there be N information nodes and K frozen action samples per node.
Use a free critic vector

$$q=(q_{1,i,k},q_{2,i,k})\in\mathbb R^{2NK}.$$

The model gives rewards R, nonnegative rows P, and discounts Gamma for each node/action.
Let `ell[i,k] = alpha(h_i) * log pi(a[i,k] | I_i)`.
For the frozen actor and quadrature,

$$
V_i(q)=\frac1K\sum_k\{\min(q_{1,i,k},q_{2,i,k})-\ell_{i,k}\},
$$
$$
(Uq)_{j,i,k}=R_{i,k}+\Gamma_{i,k}\sum_l P_{i,k,l}V_l(q),\qquad j=1,2.
$$

The action samples are drawn from the actor, with positive uniform Monte Carlo weights.
Rows are normalized in double precision before a finite solve.
True termination sets Gamma to zero and removes future entropy.
With fixed positive quadrature, the twin map contracts the joint uniform norm by `max(Gamma) < 1`.
Candidate values of both heads change inside U throughout residual recovery.
The neural target snapshot supplies an initial value argument.

This replaces v1's collapse to a single minimum-critic vector and its state/action nearest-neighbour recovery.
It also replaces the average interval entropy coefficient with the convention attached to each node's duration.

## Explicit inverse function G

Set `e(q) = q - U(q)`.
For any prescribed residual `e`, the inverse can be evaluated without freezing the continuation critic.
Define

$$
A_{il}=\delta_{il}-\frac1K\sum_k\Gamma_{i,k}P_{i,k,l},
$$
$$
b_i(e)=\frac1K\sum_k\{R_{i,k}+\min(e_{1,i,k},e_{2,i,k})-\ell_{i,k}\}.
$$

Solve `A v = b(e)` and return

$$
G(e)_{j,i,k}=R_{i,k}+\Gamma_{i,k}\sum_lP_{i,k,l}v_l+e_{j,i,k}.
$$

To verify the formula, subtract the same continuation base from both recovered heads.
Their minimum is that base plus `min(e_1,e_2)`.
Taking the frozen actor expectation gives exactly the stated equation for v.
Thus `G(e) - U(G(e)) = e`.
The averaged transition matrix has row mass below one, so A is invertible.
Distinct prescribed residuals preserve distinct recovered critic heads.

The inverse needs one N-by-N linear solve per recovery.
The implementation uses CPU double-precision NumPy/LAPACK for these small solves.
Actor, model, and critic networks use the requested Torch device.
All transfers and solves remain inside the measured update time.
The direct-solve baseline uses the same G with zero prescribed residual.

## Phase integration, cotangent, and fitting

The numerical phase is

$$
\dot e=-d e+\omega\rho,\qquad
\dot\rho=-\omega e-d\rho,\qquad
H_Q=\tfrac12(\|e\|_2^2+\|\rho\|_2^2).
$$

The midpoint update satisfies

$$
H_Q^{n+1}-H_Q^n=-\sigma d\|\bar e\|_2^2-\sigma d\|\bar\rho\|_2^2
$$

before recovery and fitting errors are added.
The implementation measures the inverse residual and its storage port, then resets e to the actual recovered residual.
It retains rho within the frozen block.
It records convergence or budget exhaustion after the declared phase count.

On an active smooth minimum branch, the value-coordinate conjugate momentum is

$$P_q=(I-DU(q))^\top\rho.$$

`residual_vjp` evaluates this function, with a deterministic branch choice at ties.
It differs from joint momentum p and from physical storage.
The finite inverse and contraction remain valid across minimum ties even where this smooth derivative needs a branch convention.

The recovered pair supervises both neural critics with stopped gradients.
The returned neural critic has a new measured residual and a fitting storage jump.
Empirical admission retains gradient-fitting progress and reports every failed complete-step check.
Strict admission requires the inverse, phase, fitting, and returned-storage bounds and otherwise restores the old critic and optimizer.
The actor update is skipped after a strict rejection.
Neither admission mode supplies a global policy-gap or physical-passivity certificate.

An actor, model, or quadrature change starts a new frozen block and resets rho.
The environment's evolution under one fixed law does not by itself change the underlying control problem.
For a common representation, the residual chart identity is

$$e_{new}(q)=e_{old}(q)-(U_{new}-U_{old})q.$$

This identity is tested numerically.
The run does not assign a cross-block monotone-storage claim to changing learned operators.

## Reward and source-target distinctions

The task reward in the simulator is the stated clipped speed reward.
The public source label is its endpoint value.
Interval targets integrate this rate in reference-time units with the same physical quadrature for every interval method.
The default start-charge entropy is `alpha_reference * h/reference_time`.
The optional discounted coefficient follows Appendix C.5.
Source critics use `alpha_CT = alpha_reference`; their interval-return counterparts use the duration-specific coefficient.

The source-equation adapters retain the duration floor and whole-correction terminal mask and log their defects.
The original archived runners retain their full published settings.
The F/J source adapters use the common observations, memory, fixed entropy, and matched case data.
These labels must remain distinct in reports.

The solver-only comparison is `interval_euler` versus `ph_sac` on the same model, actor quadrature class, node bank, reward convention, and entropy convention.
`ph_direct` checks whether a direct solve is cheaper for this finite problem.
Comparisons with `interval_sac` also change the predictor/interpolation.
Comparisons with source CT-SAC additionally change the finite target representation.
Every such comparison has separate work and error records.
