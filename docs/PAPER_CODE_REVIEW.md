# Paper-to-code review

This review compares the v3.7 code bundle (`BUNDLE_VERSION.json`, release
`3.7-mujoco-reference`) with the manuscript *Controlled SDEs, Measurements and
Verified Learning* (file `TPHT_revision.pdf`, 59 pages). Equation and section
numbers below refer to that manuscript.

Reviewed on 2026-11-20 with Python 3.13, NumPy 2.3.5, SciPy 1.17.0, MuJoCo
3.17.0, dm-control 1.0.50, Torch 2.15.0 (CPU) and Mesa EGL.

## How to reproduce

```bash
MUJOCO_GL=egl python -m pytest -q tests
MUJOCO_GL=egl python tools/paper_checks.py
```

| Check | Result |
| --- | --- |
| Bundled test suite (`tests/`) | 114 passed, 2 subtests passed |
| Independent paper checks (`tools/paper_checks.py`) | 12 PASS, 12 DIFF, 0 FAIL |

The bundled suite needs the **pinned** MuJoCo 3.17.0. With MuJoCo 4.0.1 the
four `test_mujoco_reference` replay tests fail, because the stored `.mjb`
snapshots and the replay validator are tied to the recorded MuJoCo version.
The six video/recording tests also need a working EGL library (for example
`libegl1 libgl1-mesa-dri` on Ubuntu).

## Summary

The bundle is internally consistent and well instrumented. Where the code
implements an equation from the paper, it implements it correctly. The
differences are in **experiment settings and scope**: several Table 1 / §4.7
settings differ from the paper, holds are announced to the policy, and the
paper's full bilevel structured controller (1.10) and its energy-shaping
subclass (4.26) are not implemented. The bundle's own documentation already
says this for the last two points (for example `docs/PURE_CPHT.md`: "The
manuscript's CPHT HJB/2PBVP value iteration requires a separate
implementation", and `docs/DIFFERENTIABLE_CPHT.md`: "This implementation does
not solve the unannounced-hold problem").

The manuscript also states that it "contains no new policy-training
results". The bundle agrees: `full_training_launched` and
`reference_training_launched` are both `false` in `BUNDLE_VERSION.json`.

## Verified: the code matches the paper

| Paper | Code | Evidence |
| --- | --- | --- |
| (4.31) two-tail hold law: P(h=δ)=0.891, P(h=15δ)=0.099, 0.01/13 for n=2..14 | `bench/protocol.py` `TASKS`, upstream `environment/base.py` sampler | analytic match; empirical draw from the pinned wrapper: 0.892 / 0.097 (n≈19,900) |
| (2.12) γ_h = e^{-βh} | `fjbench/agent.py`, `bench/learner.py` | direct reading |
| (2.20)+(2.22) expected entropy charge α(I)=∫α c_β(h) λ(dh) | `bench/env.py::entropy_weight` | matches to 1e-12 |
| (2.19) v⁺ = (1-ω)v + ω T̂v | `ct_picard` in `bench/operators.py`, `interval_euler` in `fjbench/finite.py` | exact identity |
| Prop. 2.1 Lipschitz factor ≤ 1-ω(1-γ_h); convergence to the fixed point | `bench/operators.py::finite_step` | measured 0.545 ≤ bound 0.985; ‖v₃₀₀₀ - v*‖ = 1e-14 |
| App. F.1 residual inverse: solve v - T̂v = e | `fjbench/finite.py::recover` (twin-critic min handled exactly) | algebra checked; validated by the solver's inverse gate |
| (4.28) exact OU wind step | `fjbench/environment.py::ou_step` | sample mean/variance match to <0.3% |
| App. A, (2.31) articulated mass matrix M(q) and bias c | `pure_cpht/mechanics.py::Robot.quantities` | equal to MuJoCo `mj_fullM` / `qfrc_bias - qfrc_passive` to 2e-16 at 25 random states |
| (2.28) gear diag(120,90,60,90,60,30), rank B = 6 < 9 | `pure_cpht/mechanics.py`, `fjbench/environment.py` (asserted at start) | checked against the compiled model |
| (2.33) Euler substep with implicit joint damping | `mujoco_reference/transition.py::capture` | prediction equals `mj_step` to 3e-14 on 20 random contact states |
| (2.26) source CT-SAC target | `ct_sac_corrected` in `bench/learner.py`, `ct_sac` in `fjbench/agent.py` | direct reading |
| (4.30) bounded action as a metric projection, not clipping | `pure_cpht/feedback.py::bounded_quadratic`, `mujoco_reference/common.py::bounded_ls` | active-set / BVLS box QP with a projected-gradient residual |
| (1.13) reward clip(v_com/g, 0, 1) with the torso subtree COM speed | `fjbench/environment.py::task_reward` | uses `subtree_linvel["torso"]` |

## Differences from the paper

Ordered by how much they affect a paper-faithful experiment.

1. **The structured controller of (1.10) / (4.26) is not implemented.** No
   code implements the energy-shaping law u = -∇U^a - K v, a = D_gear⁻¹ u, a
   learned lift L_ψ, or the local OCP (2.13) with an HJB-defect check.
   `pure_cpht` and `cpht_compare` instead learn a positive 32-parameter
   mechanical model, search 19 periodic-reference coefficients
   (CEM, gradient shooting or a neural selector), and turn that reference
   into six commands through a bounded inverse-dynamics QP. That is a
   reasonable model-based controller, but it is not the paper's bilevel
   construction. Label the results accordingly.

2. **Holds are announced to the policy.** `fjbench/models.py::InformationMap`
   appends the current requested hold `h/reference_time` to every
   actor/critic input, and the comparison runners disclose it as well. §2.6
   and §4.7 require holds to be unannounced before the action ("Keep all
   future holds unannounced in the matched experiment"), with the backup
   averaged over λ(dh | I) as in (2.22).

3. **J-wind applies its force at a foot.** `bench/env.py` picks
   `force_body` as the deepest, heaviest body (`bfoot`), with a 5 cm offset.
   `fjbench/environment.py` reuses that point for wind. (A.22) puts the wind
   force at the torso centre of mass, with no direct leg entries.

4. **J-wind has no wind sensor.** `observation_dim` is 17 for J-wind.
   Table 1 gives 18 channels, including o^w = w + ε, ε ~ N(0, 1 N²).

5. **Table 1 / §4.7 parameter ranges differ.**

   | Setting | Paper | Code default (`fjbench/config.py`) |
   | --- | --- | --- |
   | J-static friction | Unif[0.6, 1.0] | {0.6, 1.0, 1.4} (a multiplier on every geom's sliding friction) |
   | J-static payload | Unif[0, 2] kg | {0, 1, 2} kg |
   | J-static speed threshold g | Unif[6, 10] m/s | {6, 10, 14} |
   | J-wind σ_w | 2 N s^-1/2 | 0.4 × force scale 3 = 1.2 N s^-1/2 |
   | Transfer tests | friction [0.3, 0.5], payload [3, 4] kg, g = 12, dw = -2w dt + 4 dW | friction {0.7, 1.3}, payload {0.25, 1.75}, g {7, 13}, κ = 0.5, σ = 0.7 |
   | F-position noise | 0.005 m root height, 0.01 rad angles | 0.01 on all eight channels |

6. **Payload inertia.** The code scales the torso inertia by
   (m+m_load)/m (Iyy 1.169 at 2 kg). The note after (A.10) adds the inertia
   of a 5 cm payload at the torso centre, I_T + (2/5) m_load (0.05)²
   (Iyy 0.888 at 2 kg).

7. **Default entropy convention.** The default `entropy_convention="start"`
   uses α_h = α·h. (2.20) prescribes α_h = α c_β(h). The `"discounted"`
   option implements (2.20). Both satisfy α_h/h → α, but they define
   different finite-hold operators. Set `entropy_convention: "discounted"` for
   a paper-faithful run.

8. **App. F.1 numerical oscillator.** `bench/operators.py::phase_coefficients`
   is the exact implicit-midpoint map of ė = -d e + ω ρ, ρ̇ = -ω e - d ρ
   (damping on both coordinates, frequency ω = 3). App. F.1 states
   ė = ρ, ρ̇ = -e - dρ (damping on ρ only, unit frequency). The code's
   ledger check (`midpoint_balance_defect`) matches its own oscillator, so
   this is a definition mismatch, not an arithmetic error. It only affects
   the optional solver ablation (comparison 3 in §4.7).

9. **Reward quadrature (2.34).** The code integrates e^{-βu} exactly over each
   substep and weights the right-endpoint reward with exponent jδ for
   j = 0..N-1. (2.34) uses δ e^{-βjδ} for j = 1..N. The difference is O(βδ),
   about 0.2% for a 15-substep hold. Both are consistent first-order
   quadratures, but matched comparisons should name one.

## Not checked

- Full training runs and the published numerical results. They need a GPU
  server and the user's `sensor/` run directory.
- The contact solver in `pure_cpht/contact.py` (a learned finite-iteration
  Coulomb-cone QP). It was checked only through the bundled tests. It is a
  surrogate and does not reproduce MuJoCo's soft-contact problem (2.32).
- `physical_fj`, `environment_belief` and the CT-terrain adapter. They were
  checked only through the bundled tests.

## Other observations

- `vendor/ct-rl-source.zip` (the pinned upstream CT-RL source) contains no
  license file. Check that redistribution is allowed before making this
  repository public.
- `requirements-physical-first.txt` pins `torch==2.14.1`. The review used
  Torch 2.15.0. `dm-control` 1.0.50 depends on `labmaze`, which has no
  Python 3.13 wheel. Use Python 3.11/3.12, or install `dm-control` with
  `--no-deps`; the cheetah suite does not need `labmaze`.
