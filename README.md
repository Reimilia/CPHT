# CPHT / TPHT: structured control learning on the DMC cheetah

This repository holds the reference implementation and benchmark harness for
the manuscript **Controlled SDEs, Measurements and Verified Learning** (the
"TPHT revision"). The manuscript formulates control learning as a bilevel
problem: an outer learner chooses a phase-space lift, port-Hamiltonian (pH)
dynamics and a realizable command decoder, and an inner local optimal-control
problem runs in physical time. Its cheetah examples compare three ways of
predicting the next interval:

- **CT**: model-free continuous-time RL (CT-SAC / CT-TD3) from measured transitions.
- **CP (CPHT)**: a nominal, deterministic learned pH rollout.
- **TP (TPHT)**: a stochastic or ensemble pH rollout.

All three use the same forward-Euler value update, v⁺ = v + τ Ĥ_S[v].

The code is release **v3.7** (`BUNDLE_VERSION.json`). It contains the matched
CT/CP/TP learners, actor-free learned-mechanics controllers, sensing
ablations, terrain/contact benchmarks, and a no-training MuJoCo reference
comparison with exact transition replay. **No full policy-training results
are included**, as in the manuscript. Short smoke runs check that the code
runs. They do not rank methods.

> **Read [`docs/PAPER_CODE_REVIEW.md`](docs/PAPER_CODE_REVIEW.md) before
> running an experiment for the paper.** Where the code implements a paper
> equation, it implements it correctly. Several experiment settings and one
> protocol rule differ from the manuscript, and the full bilevel structured
> controller (1.10)/(4.26) is not implemented yet.

## Architecture

![Architecture overview](docs/architecture/cpht-architecture.png)

Open [`docs/architecture/cpht-architecture.html`](docs/architecture/cpht-architecture.html)
in a browser for the interactive version. It was generated with
[Archify](https://github.com/tt-a1i/archify) from this repository's source,
and every box links to the file and lines it summarizes. Its JSON source is
`docs/architecture/cpht-architecture.json`.

## Where the paper lives in the code

| Paper | Code |
| --- | --- |
| §1.5, Table 1: cheetah task, F/J cases, observations | `fjbench/config.py`, `fjbench/environment.py` |
| §2.2, (2.8), (4.31): physics substep, held actions, two-tail hold law | `bench/protocol.py`, `bench/env.py` (pinned CT-RL wrapper in `vendor/`) |
| (2.12), (2.19), (2.24): common backup, Euler value step, critic fit | `fjbench/finite.py`, `fjbench/agent.py` (`interval_euler`), `bench/operators.py` (`ct_picard`) |
| (2.26): source CT-SAC / CT-TD3 targets | `fjbench/agent.py` (`ct_sac`, `ct_td3`), `bench/learner.py` |
| App. F.1: residual-coordinate pH value solver | `fjbench/finite.py` (`ph_sac`, `ph_direct`), `bench/operators.py` |
| App. A, (2.3), (2.31): articulated mechanics, H = ½pᵀM⁻¹p + U | `pure_cpht/mechanics.py`, `physical_fj/mechanics.py` |
| (2.32), §3.6, App. B/C: contact | `pure_cpht/contact.py`, `fjbench/contact.py`, `fjbench/terrain.py` |
| (2.33)–(2.35): held MuJoCo transition, implicit damping, replay | `mujoco_reference/transition.py` |
| §3.1, §4.7: sensing and belief assimilation | `environment_belief/` |
| §4.1, §4.6: calibration and parameter sensitivities | `pure_cpht/learning.py`, `physical_fj/prepare.py`, `fjbench/calibration.py` |
| (4.28): exact OU wind step | `fjbench/environment.py::ou_step` |
| Differentiable model-based control (BPTT, CEM, shooting, selector) | `cpht_compare/` |

## Install

Use Python 3.11 or 3.12 with a Torch build that fits your hardware:

```bash
python -m pip install -r requirements-physical-first.txt
```

Keep the pinned `mujoco==3.17.0`. The stored `.mjb` replay snapshots and the
transition validator are tied to that version. For headless rendering, set
`MUJOCO_GL=egl` and install a system EGL library (on Ubuntu:
`libegl1 libgl1-mesa-dri`). On Python 3.13, `labmaze` has no wheel. Install
`dm-control==1.0.50` with `--no-deps`; the cheetah suite does not need
`labmaze`.

On first use, `bench/protocol.py` extracts the hash-checked CT-RL source to
`vendor/extracted/`. It refuses to run if that copy has been edited.

## Check the installation

```bash
MUJOCO_GL=egl python -m pytest -q tests
```

This runs 114 tests in about 2 minutes on 4 CPU cores. It needs no GPU and
no training data.

```bash
MUJOCO_GL=egl python tools/paper_checks.py
```

This checks the manuscript's equations against the code and MuJoCo. Each line
is PASS, DIFF (a different convention or setting from the paper) or FAIL.

```bash
python -c "from mujoco_reference.transition import validate; print(validate('validation/reference_samples/fixed/F_transition.json'))"
```

This replays a recorded real cheetah transition bit-for-bit and checks the
force balance M q̈ + c = τ + J_cᵀ f.

## Run experiments

Every launcher is dry by default: it prints its plan, and runs only with
`DRY_RUN=0`.

| Experiment | Launcher | Guide |
| --- | --- | --- |
| Differentiable CPHT comparison (CEM / shooting / selector / selector+value) | `scripts/run_cpht_comparison.sh` | [docs/DIFFERENTIABLE_CPHT.md](docs/DIFFERENTIABLE_CPHT.md) |
| No-training MuJoCo reference controllers and transition samples | `scripts/run_mujoco_reference.sh` | [docs/MUJOCO_REFERENCE.md](docs/MUJOCO_REFERENCE.md) |
| Actor-free learned-mechanics feedback (`pure_cpht`) | `scripts/run_pure_cpht.sh` | [docs/PURE_CPHT.md](docs/PURE_CPHT.md) |
| Physical-feature SAC / CT-SAC ablation | `scripts/run_physical_first.sh` | [docs/PHYSICAL_FIRST.md](docs/PHYSICAL_FIRST.md) |
| Sensing / belief ablations | `scripts/run_sensing.sh` | [docs/SENSING_EXPERIMENT.md](docs/SENSING_EXPERIMENT.md) |
| Matched CT / CP / TP learners on F and J | `scripts/run_F.sh`, `scripts/run_J.sh`, `scripts/run_case.sh` | [docs/DESIGN_FJ.md](docs/DESIGN_FJ.md) |
| Terrain, contact and path-speed tasks | `scripts/run_terrain.sh`, `scripts/run_contact.sh`, `scripts/run_path_speed.sh` | [docs/CONTACT_BENCHMARK.md](docs/CONTACT_BENCHMARK.md), [docs/PATH_SPEED_PROTOCOL.md](docs/PATH_SPEED_PROTOCOL.md), [docs/TERRAIN_ALIGNMENT.md](docs/TERRAIN_ALIGNMENT.md) |

For example, an inexpensive end-to-end smoke run of the newest comparison:

```bash
PROFILE=cases/cpht_comparison_smoke.json \
RUN_ROOT=runs/cpht_smoke ARTIFACTS=artifacts/cpht_smoke \
REPORT=reports/cpht_smoke.json \
STAGE=all DRY_RUN=0 DEVICE=cpu bash scripts/run_cpht_comparison.sh
```

For a paper-faithful matched run, set `"entropy_convention": "discounted"`
in the case file and review the settings in
[`docs/PAPER_CODE_REVIEW.md`](docs/PAPER_CODE_REVIEW.md#differences-from-the-paper).

## Repository layout

```text
bench/                 CT-RL protocol, hash-checked upstream wrapper, legacy matched learners
fjbench/               F/J cases, environments, matched CT/CP/TP agents, finite solvers, contact/terrain
physical_fj/           instrumented physical calibration and physical-feature SAC
pure_cpht/             learned pH mechanics, contact closure, periodic-reference feedback (no actor/critic)
cpht_compare/          differentiable comparison: imagination rollout, CEM, shooting, selector, value
environment_belief/    causal terrain/material belief from sensors
mujoco_reference/      MuJoCo reference controllers and exact transition replay audits
cases/, configs/       experiment presets
scripts/               dry-by-default launchers (local and SLURM)
tests/                 unit and integration tests
tools/paper_checks.py  paper-equation cross-check
validation/            recorded validation evidence (reports, gradient checks, replay samples)
prechecks/             small model fixture used by tests and previews
vendor/                pinned CT-RL source archive (hash-checked)
third_party/           SuperMimic-Reproduced attribution and MIT license
docs/                  current guides, the review, the architecture diagram
docs/history/          earlier READMEs, validation notes and changelogs (v1 to v3.7)
*.py (root)            command-line entry points (prepare, train, eval, render, aggregate)
```

`apply_calibration_fix.py` and `apply_finite_fix.py` are one-off installers
that patch older (v2/v3.1) checkouts. This tree already includes both fixes.

## Validation evidence

`validation/` keeps the evidence that shipped with v3.7: unit-test logs,
gradient checks, smoke and half-second comparison reports, four portable
transition samples with their compiled models, and the full-course MuJoCo
reference results. The full-course reference comparison ran 20 episodes
without training. In all of them, the 12 s course timed out before
completion; see `validation/mujoco_reference/report.md`.

## License

GPL-3.0 (see `LICENSE`). `third_party/supermimic_reproduced/` keeps the MIT
notice for the adapted SuperMimic-Reproduced patterns.
`vendor/ct-rl-source.zip` is the pinned upstream CT-RL source. It ships
without a license file, so confirm that redistribution is allowed before
publishing this repository.
