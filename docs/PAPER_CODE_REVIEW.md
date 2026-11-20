# Paper-to-code review

This review compares the code with the manuscript *Controlled SDEs,
Measurements and Verified Learning* (file `TPHT_revision.pdf`, 59 pages).
Equation and section numbers refer to that manuscript. It was first done on
the v3.7 bundle and updated for v3.8, which reduced the repository to one
experiment pipeline (see [EXPERIMENTS.md](EXPERIMENTS.md)).

Checked on 2026-11-20 with Python 3.13, NumPy 2.3.5, SciPy 1.17.0, MuJoCo
3.17.0, dm-control 1.0.50, Torch 2.15.0 (CPU) and Mesa EGL.

## How to reproduce

```bash
MUJOCO_GL=egl python -m pytest -q tests
./run_experiments.sh smoke verify      # replays + tools/paper_checks.py
```

| Check | v3.7 | v3.8 |
| --- | --- | --- |
| Test suite (`tests/`) | 114 passed | 65 passed (tests of removed modules dropped; pipeline tests added) |
| Paper checks (`tools/paper_checks.py`) | 12 PASS, 12 DIFF, 0 FAIL | 13 PASS, 10 DIFF, 0 FAIL |
| Smoke experiment, all stages | not part of v3.7 | 10 runs, 40 paired evaluations, 0 errors |

Use the **pinned** MuJoCo 3.17.0: with 4.0.1 the transition-replay tests fail,
because the stored `.mjb` snapshots are tied to the recorded version. Video
recording needs a working EGL library (`libegl1 libgl1-mesa-dri` on Ubuntu).

## Summary

Where the code implements an equation from the paper, it implements it
correctly. The remaining differences are in scope and settings:

- The paper's full bilevel structured controller (1.10) and its
  energy-shaping subclass (4.26) are not implemented. The pH model is used as
  a calibrated predictor that a gait search or a neural selector optimizes
  through.
- Several Table 1 variants (J-static, J-wind, F-position) are defined in
  `fjbench/config.py` with settings that differ from the paper.
  `run_experiments.sh` does not run them: its F and J are the terrain-course
  cases.
- v3.8 fixed the announced-hold deviation (see below).

The manuscript states that it "contains no new policy-training results". The
repository likewise contains no full training results; the smoke scale only
checks that every stage runs.

## Verified: the code matches the paper

| Paper | Code | Evidence |
| --- | --- | --- |
| (4.31) two-tail hold law: P(h=δ)=0.891, P(h=15δ)=0.099, 0.01/13 for n=2..14 | `bench/protocol.py` `TASKS`, upstream `environment/base.py` sampler | analytic match; empirical draw from the pinned wrapper: 0.892 / 0.097 (n≈19,900) |
| (2.12) γ_h = e^{-βh} | `physical_fj/learner.py` (CT-SAC), `cpht_compare/rollout.py` | direct reading |
| (2.20)+(2.22) expected entropy charge α(I)=∫α c_β(h) λ(dh) | `bench/env.py::entropy_weight` | matches to 1e-12 |
| (2.19) v⁺ = (1-ω)v + ω T̂v, and Prop. 2.1 | `bench/operators.py::finite_step` (`ct_picard`), a reference implementation | exact identity; Lipschitz factor 0.545 ≤ bound 0.985; ‖v₃₀₀₀ - v*‖ = 1e-14 |
| (4.28) exact OU wind step | `fjbench/environment.py::ou_step` | sample mean/variance match to <0.3% |
| App. A, (2.31) articulated mass matrix M(q) and bias c | `pure_cpht/mechanics.py::Robot.quantities` | equal to MuJoCo `mj_fullM` / `qfrc_bias - qfrc_passive` to 2e-16 at 25 random states |
| (2.28) gear diag(120,90,60,90,60,30), rank B = 6 < 9 | `pure_cpht/mechanics.py`, `fjbench/environment.py` (asserted at start) | checked against the compiled model |
| (2.33) Euler substep with implicit joint damping | `mujoco_reference/transition.py::capture` | prediction equals `mj_step` to 3e-14 on 20 random contact states |
| (2.26) source CT-SAC rate target | `physical_fj/learner.py` (the `ct_sac` baseline) | direct reading |
| (4.30) bounded action as a metric projection, not clipping | `pure_cpht/feedback.py::bounded_quadratic`, `mujoco_reference/common.py::bounded_ls` | active-set / BVLS box QP with a projected-gradient residual |
| (1.13) reward speed is the torso subtree COM speed | `fjbench/environment.py::task_reward` | uses `subtree_linvel["torso"]` |
| §2.6, §4.7 unannounced holds (fixed in v3.8) | `physical_fj/mechanics.py::announced_hold`, `cpht_compare/pipeline.py` | on the irregular clock the policy sees only 0.01 s while realized holds vary over 0.002–0.03 s |

## Fixed in v3.8: holds were announced

In v3.7 the observation vector carried the current requested hold
(`raw[30]`), and the runners passed it to the controller as the first
imagined interval. The paper requires holds to be unannounced before the
action (§2.6, §4.7). In v3.8 the `hold_disclosure` setting defaults to `false`:

- `PhysicalEnvironment.observe` writes the nominal hold into `raw[30]`.
- Controllers receive `announced_hold()`.
- Imagined replays start from the nominal hold.

`--disclose-holds` restores the old interface as an ablation. Checkpoints
saved before v3.8 have no such key and keep the interface they were trained
with.

## Remaining differences from the paper

1. **The structured controller of (1.10) / (4.26) is not implemented.** No
   code implements the energy-shaping law u = -∇U^a - K v, a = D_gear⁻¹ u, a
   learned lift L_ψ, or the local OCP (2.13) with an HJB-defect check. The pH
   methods learn a positive 32-parameter mechanical model and then search
   (CEM, gradient shooting) or learn (selector, selector + value) 19
   periodic-reference coefficients through it. A bounded six-motor QP turns
   the reference into commands. This is model-based learning with a pH
   predictor, but it is not the paper's bilevel construction.

2. **Table 1 variants are not run, and their settings differ.**
   `fjbench/config.py` still defines F-position, J-static and J-wind, but
   `run_experiments.sh` uses the terrain-course F/J cases. For a Table 1
   replication these settings would need to change:

   | Setting | Paper | Code (`fjbench/config.py`) |
   | --- | --- | --- |
   | J-static friction | Unif[0.6, 1.0] | {0.6, 1.0, 1.4} (a multiplier on every geom's sliding friction) |
   | J-static payload | Unif[0, 2] kg | {0, 1, 2} kg |
   | J-static speed threshold g | Unif[6, 10] m/s | {6, 10, 14} |
   | J-wind σ_w | 2 N s^-1/2 | 0.4 × force scale 3 = 1.2 N s^-1/2 |
   | J-wind sensor | 18 channels, o^w = w + ε | 17 channels, no wind channel |
   | J-wind force point (A.22) | torso centre of mass | the `bfoot` body, 5 cm offset |
   | Payload inertia (A.10) | I_T + (2/5) m_load (0.05)² | torso inertia × (m+m_load)/m |
   | Transfer tests | friction [0.3, 0.5], payload [3, 4] kg, g = 12, dw = -2w dt + 4 dW | friction {0.7, 1.3}, payload {0.25, 1.75}, g {7, 13}, κ = 0.5, σ = 0.7 |
   | F-position noise | 0.005 m root height, 0.01 rad angles | 0.01 on all eight channels |

3. **Reward quadrature (2.34).** The environment integrates e^{-βu} exactly over
   each substep and weights the right-endpoint reward with exponent jδ for
   j = 0..N-1. (2.34) uses δ e^{-βjδ} for j = 1..N. The difference is O(βδ),
   about 0.2% for a 15-substep hold. All methods share the same convention.

4. **App. F.1 numerical oscillator.** `bench/operators.py::phase_coefficients`
   is the exact implicit-midpoint map of ė = -d e + ω ρ, ρ̇ = -ω e - d ρ
   (damping on both coordinates, ω = 3). App. F.1 states ė = ρ,
   ρ̇ = -e - dρ. This reference implementation is checked by the tests but not
   used by the experiment runner.

The v3.7 finding about the default finite-hold entropy convention (2.20)
concerned the matched interval learners, which v3.8 removed. The CT-SAC
baseline uses the source rate convention (2.26).

## Not checked

- Pilot- and full-scale training results. They need hours of compute (see
  [EXPERIMENTS.md](EXPERIMENTS.md)).
- The contact closure in `pure_cpht/contact.py` (a learned finite-iteration
  Coulomb-cone QP). It was checked only through the tests. It is a surrogate
  and does not reproduce MuJoCo's soft-contact problem (2.32).

## Other observations

- `vendor/ct-rl-source.zip` (the pinned upstream CT-RL source) contains no
  license file. Check that redistribution is allowed before making this
  repository public.
- `dm-control` 1.0.50 depends on `labmaze`, which has no Python 3.13 wheel.
  Use Python 3.11/3.12, or install `dm-control` with `--no-deps`; the cheetah
  suite does not need `labmaze`.
