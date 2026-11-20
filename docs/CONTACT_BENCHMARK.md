# Terrain and material contact benchmark

## 1. State, map, and contact computation

The contact extension follows Algorithms 3–4, Section A.4, and Appendix C of the attached measurements note.
`cases/sources.json` records the exact supplied document hash.
The new numerical map ranges are explicit benchmark choices.
Existing v2 wind and payload presets retain their earlier settings.

Let a static map be

$$
m=\{\phi_m(s),\mu_m(s),\tau_m(s),\zeta_m(s)\},
\qquad X_t=(q_t,p_t,m),\qquad dm_t=0.
$$

Here $s$ is world horizontal position, $\phi_m$ is elevation, $\mu_m$ is sliding friction,
$\tau_m$ is the positive MuJoCo contact time constant in seconds, and $\zeta_m$ is its dimensionless damping ratio.
The last two parameters specify a soft-contact response. They require calibration before a physical material claim.

At each physics step, collision detection finds the active set $\mathcal C(q;m)$.
Its points, normal directions, selected materials, and Jacobians depend on the current articulated pose.
The mechanical update has the form

$$
\dot q=M(q)^{-1}p,\qquad
\dot p=-\partial_q H(q,p)-D(q)\partial_pH(q,p)+B(q)a
       +\sum_{c\in\mathcal C(q;m)}J_c(q)^\top F_c.
$$

MuJoCo solves the coupled contact problem to obtain $F_c$.
Each back-foot point Jacobian has direct support on the three root coordinates and the three back-leg joints.
Each front-foot point Jacobian has direct support on the roots and front-leg joints.
The non-diagonal inertia couples the resulting accelerations across the robot.
`ContactAudit` records each force, its current Jacobian pullback, and the directly affected coordinates.
Other bodies can also touch the ground; their contacts have their own Jacobians.

F fixes the complete map across all episodes and training seeds. Robot reset positions and joint angles can vary.
J samples a map from a finite declared bank at episode start and retains it for that episode.
The queried local environment changes as the robot moves through this map.
The changing contact graph belongs to both cases.
Map sampling between episodes defines the J family.
This revision implements static surfaces. A translating obstacle or a surface changed by a timed event requires its own state and transition law.

## 2. Surface construction and synthetic materials

The default track has 240 one-metre cells, extruded 1.5 m to each side of the motion plane.
Every cell is a closed convex mesh wedge.
Adjacent cells share their top edge exactly.
The default elevation is a bounded two-frequency profile with a flat initial pad.
Its sampled training amplitude lies in $[0.08,0.24]$ m and its base wavelength lies in $[5,9]$ m.
The held-out elevation range uses amplitude $[0.30,0.40]$ m and wavelength $[3.5,4.5]$ m.
The profile is piecewise linear. Its normal can change at an edge.
Endpoint pads extend the track; the original plane has zero collision masks.

The training sliding coefficient lies in $[0.35,1.0]$ and the friction transfer range lies in $[0.12,0.25]$.
The contact time constant lies in $[0.02,0.05]$ s and the damping ratio lies in $[0.8,1.3]$.
The contact-response transfer uses $[0.06,0.08]$ s and $[0.45,0.65]$.
Parameters are constant in two-cell blocks.
All patches use `condim=3` and `solimp=(0.9,0.95,0.001,0.5,2)`.

Ground geoms have priority 2; the robot geoms retain priority 0.
MuJoCo therefore uses the ground's friction and contact response for each dynamic pair.
This makes a low-friction patch effective even when the foot has a larger coefficient.
The audit compares `data.contact[i].friction[0]` with the selected ground coefficient at every physics step.
It also checks the effective contact time and damping against the selected patch.
It also records penetration and friction utilization from the solved forces.
A force utilization or point speed is a diagnostic; classifying stick/slip requires a stated tolerance and a contact-law interpretation.

Use four profiles to isolate factors:

- `elevation`: varying height, uniform friction and contact response.
- `friction`: flat height, varying friction, uniform contact response.
- `material`: flat height, varying friction and contact response.
- `mixed`: all three physical factors.

The contact model retains six motor commands and the original robot mass and gears.
J/contact keeps global friction scaling and payload fixed by default.
Set `contact_robot_variation=true` for the separate combined robot-parameter study.
That option reuses the v2 global friction and payload grids in addition to the local map.

## 3. Permitted environmental information

Let the permitted history be

$$
\mathcal H_k=\sigma(o^{\rm robot}_{0:k},o^{\mathcal E}_{0:k},a_{0:k-1},r_{0:k-1},h_{0:k}),
\qquad I_k=(g,\operatorname{GRU}_{\psi}(\mathcal H_k),h_k).
$$

The known speed target $g$ remains a task descriptor.
The map $m$ and its layout identifier remain backend truth.
The shared observer uses measured histories. No supervised latent map label enters its loss.
The learned positive operator takes $I_k$, the common current action, and the known task stratum.
Environmental effects therefore enter both the policy and the value backup through the same permitted information.

The geometry sensor samples seven offsets from the torso: $(-1,-0.5,0,0.5,1,2,3)$ m.
At each offset it returns elevation relative to the torso and the local slope.
Height noise is independent Gaussian with standard deviation 0.005 m; slope noise has standard deviation 0.01.
A sample arrives at reset and after each decision, with zero latency and zero acquisition fee.
The sample available at decision start is the only current environmental sample used by the actor.
The simulator uses the true current position to implement this local sensor.
It supplies neither world position nor a full-map lookup to the network.

Native proprioception has 17 components. The default geometry sensor adds 14, for a 31-component observation.
The `history` ablation uses 17 and infers contact effects from proprioceptive, command, reward, and timing history.
The `material` information experiment adds local noisy values of $\mu$, $\tau$, and $\zeta$ to the geometry sensor, for 52 components.
Their noise standard deviations are 0.03, 0.002 s, and 0.03.
This is a declared synthetic material sensor. It enlarges the policy information available to every method in that experiment.

RGB colours, map indices, exact material arrays, contact forces, and Jacobian audits stay outside replay features.
Checkpoint metadata and audit files can retain those quantities for verification.
Changing a sensor changes the calibration identity and requires a fresh shared observer.
The finite GRU history supplies an approximate information state; the code reports predictive closure diagnostics.
A local terrain scan can leave distant regions and contact parameters uncertain.
Exact posterior sufficiency and a global physical safety certificate need further assumptions and validation.

## 4. Shared preparation and the training algorithm

Both cases use random spawn positions in $\{0,4,8,12,16\}$ m on their allowed maps.
This reset law exposes slopes and multiple material patches during random-action calibration.
The source random-joint initialization and its settling procedure remain in use.
Settling contributes to wall-clock acquisition cost. Decision/physics ledgers count the subsequent recorded transitions.

F has one full-difficulty map and a stationary reset law.
J has 32 training layouts, 8 validation layouts, and 16 held-out layouts in disjoint seed namespaces.
A map identity includes the physical fields and difficulty.
The preparation records the realized calibration layouts, difficulties, task descriptors, and spawn positions in an audit list.
Inspect that list before a large run. Random-action data can still have limited state coverage.

The default J curriculum uses difficulty $0.35$ during the first 15% of policy decisions,
$0.65$ from 15% to 40%, and $1$ for the remaining 60%.
Height amplitude is multiplied by difficulty.
Friction and contact response interpolate from a uniform firm surface toward the sampled map.
The stage is fixed during an episode.
Preparation cycles through all three difficulty levels and known task strata.
Evaluation uses full difficulty.
Use `cases/contact_no_curriculum.json` for a fixed full-range reset distribution throughout training.
A curriculum changes the training reset prior. Its benefit is an empirical ablation and supplies no new stationary convergence theorem.

```text
Inputs: case F or J; permitted sensor; map family; known task family;
        paired seed; interaction budget N; common preparation budget Ncal;
        positive operator architecture; solver settings; curriculum schedule.
Output: frozen policy and observer; audit logs; independent evaluation records.

1. Freeze the physical family, reward, motor limits, observation law, and hold rule.
2. Collect Ncal measured decisions with the common random-action controller.
   F: reuse its one map. J: cycle calibration difficulty and sample training maps.
   Randomize the declared spawn positions. Keep episode-level truth in audits.
3. Split calibration records by episode. Fit the shared history observer on
   training records and test its observation prediction and closure diagnostics.
   Freeze its parameters. Exclude validation episodes from every later gradient.
4. Fit one positive operator member for F, or the declared ensemble for J.
   Condition transition support and reward prediction on the known task g.
   Share this preparation artifact across all compared methods at the same seed.
5. For each policy episode, select the curriculum stage by physical-decision
   budget, draw the allowed map/reset, and retain the map until episode end.
6. At decision k, encode only the available history. Supply the queued hold h_k.
   Draw the six-motor action. Run collision detection/contact solve at each
   physics substep; integrate reward and contact diagnostics over the hold.
   Append only the permitted measured transition to replay.
7. Freeze observer, model, actor, task weights, entropy convention, nodes,
   and action quadrature for one critic block. Apply the selected CT target,
   interval backup, Euler update, pH residual solve, or direct finite solve.
   Fit both critics. Apply the declared numerical admission rule.
   Then update actor/targets. Charge solver, fitting, sensing, and rejected work.
8. Continue until N decisions, including Ncal. Save replay, optimizers, RNG,
   counters, and next episode index for recovery at an episode boundary.
9. Freeze learning. Evaluate matched policies on paired seeds at full difficulty,
   with the same measurements, motor limits, hold clocks, and reset distribution.
   Report control score, contact diagnostics, and training-seed uncertainty.
```

The default freezes both observer and operator during policy comparison.
The optional `cases/contact_online_operator.json` updates the operator from permitted replay.
Its candidate update must pass the same held-out calibration score rule used in v2.
The observer and node bank stay fixed.
Learned sensing, learned holding times, and full online observer replacement define separate studies.
All methods retain the same exogenous holding-time rule in this comparison.

## 5. Matched comparison and held-out tests

Compare CT-SAC/CT-TD3 source-equation adapters, measured interval SAC, learned-model SAC,
interval Euler, the pH residual update, and the direct finite solve using the same permitted information.
`DESIGN_FJ.md` defines the common frozen twin-critic operator and the exact recovery map.
The three finite solvers share that operator, node bank, action quadrature, reward, and task support.
Interaction budgets match. Report inner solver and fitting costs separately.
The default empirical fitting mode retains regression progress and reports numerical storage debts.
Use strict admission as a separate numerical experiment with its rejections and added cost visible.

J/contact evaluation separates the following interventions:

- `iid`: new resets/sensor streams on the training layout bank, at full difficulty.
- `layout`: disjoint map seeds with training parameter ranges.
- `terrain`: paired training layout seed with changed amplitude and wavelength only.
- `friction`: paired layout and geometry with held-out sliding coefficients only.
- `material`: paired layout, geometry, and friction with held-out time/damping parameters only.
- `contact`: held-out layout plus all three contact/environment factors.
- `task`: the v2 held-out speed descriptors on the training physical map family.
- `joint`: contact transfer plus held-out speed; also robot-regime transfer when explicitly enabled.

`regime` is available when `contact_robot_variation=true`.
A profile must contain a factor before its single-factor test is meaningful.
A material-response test on an elevation-only profile raises an error.
F/contact evaluates its own fixed map with paired randomized resets.
Use J for changed-map transfer.

Run checkpoint/settings selection only on the `validation` split.
It uses a disjoint validation layout bank with training parameter ranges.
That bank is a policy-validation set. Calibration validation remains a held-out episode split of acquired training-map data.
Report the same chosen checkpoint rule for all methods before test evaluation.
Held-out task evaluation uses the frozen actor; the finite training-node operator supplies no new-task value certificate.

Primary score remains the normalized time-integrated forward-speed reward from v2.
Add net horizontal distance, low-torso time fraction, summed contact slip, foot-specific normal impulses,
maximum penetration, effective coefficient error, and contact-point work.
Low-torso time uses torso clearance below 0.25 m and keeps the original reward/termination rule.
dm-control can update contact geometry after a step while retaining earlier acceleration-stage forces.
The audit copies `MjData` and runs an endpoint forward solve on that copy before reading force and Jacobian.
This keeps each diagnostic force aligned with its contact point and leaves trajectory/warm-start arrays unchanged.
The ledger counts these extra audit solves; wall-clock acquisition and rollout cost includes them.
Contact-point work and normal-force integrals use this endpoint quadrature; they are numerical diagnostics.
It supplies no discrete passivity proof or mechanical safety guarantee.
Average within known task first, then across independent training seeds.
Retain paired seed differences and bootstrap over training seeds.

## 6. Environment sampling and headless prechecks

`sample_envs.py` writes complete hashed backend maps and optional profile PNGs.
Training reproduces maps from the same config and seed sampler; it does not ingest an actor-accessible manifest.
`preview_envs.py` replays the exact saved map and writes MP4 plus contact audit JSON.
The render backend is selected before simulator import.
The default fixed preview clock is 0.01 s; training keeps its declared irregular clock.
The preview records simulation/frame timestamps, so duration quantization remains visible.

Use zero/random/sine controls for a cheap physics check.
Use the frozen checkpoint option for a later policy inspection.
Tour mode moves the camera only. Its contact and penetration figures describe no physical trajectory.
The output labels distinguish camera tours from rollouts.

Before training, inspect that surfaces join, the robot starts above the selected patch,
feet contact the expected materials, effective coefficients match, and penetration stays acceptable
under the declared soft-contact model and physics step.
A timestep refinement study remains necessary for trajectory accuracy and energy claims.
Inspect normal/tangential force and Jacobian support when front and back feet touch different patches.
The optional simulator tests verify that a back-foot force has zero direct front-leg entries and vice versa.

## 7. Source and implementation references

- Attached measurements note, Algorithms 3–4, Section A.4, and Appendix C; exact bytes in `cases/sources.json`.
- [MuJoCo contact parameter combination rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
- [MuJoCo mesh and geom XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html).
- [MuJoCo native Python API](https://mujoco.readthedocs.io/en/stable/python.html).
- [dm-control rendering backends](https://github.com/google-deepmind/dm_control#rendering).
- `fjbench/contact.py`: map sampler, sensors, MJCF wedges, and contact/Jacobian audit.
- `fjbench/environment.py`: backend map, causal measurements, and physics hook.
- `fjbench/calibration.py`: exposure distribution and shared information/operator fitting.
- `fjbench/runner.py`: decision-budget curriculum and unchanged frozen critic blocks.
- `fjbench/evaluation.py`: paired transfer tests and contact diagnostics.
