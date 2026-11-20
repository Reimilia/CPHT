"""Independent numerical checks of TPHT_revision.pdf equations against the code.

Run from the repository root:  MUJOCO_GL=egl python tools/paper_checks.py

Each check prints PASS (code matches the paper), DIFF (the code deliberately or
accidentally follows a different convention or setting than the paper), FAIL
(the code is inconsistent with its own stated equation) or ERROR. Equation and
section numbers refer to TPHT_revision.pdf. The script only reads the code and
runs MuJoCo; it trains nothing and writes nothing.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
results = []


def report(name, status, detail):
    results.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


# ---------------------------------------------------------------- (4.31) holds
def check_hold_law():
    from bench.env import entropy_weight
    from bench.protocol import TASKS
    t = TASKS["cheetah-run"]
    n = np.arange(1, 16)
    p = np.zeros(15)
    p[0], p[-1] = t["tail_p"] * t["tail_split"], t["tail_p"] * (1 - t["tail_split"])
    p[1:-1] = (1 - t["tail_p"]) / 13
    ok = abs(p[0] - .891) < 1e-12 and abs(p[-1] - .099) < 1e-12 and np.allclose(p[1:-1], .01 / 13)
    report("(4.31) two_tail_uniform hold probabilities", "PASS" if ok else "FAIL",
           f"P(h=d)={p[0]:.3f}, P(h=15d)={p[-1]:.3f}, P(n=2..14)={p[1]:.6f}")
    # (2.20)/(2.22): alpha(I) = E_h[alpha * c_beta(h)], time in native 0.01 s units
    beta = -math.log(.98)
    expected = float(p @ (-np.expm1(-beta * n * .002 / .01) / beta))
    got = entropy_weight("cheetah-run", .01, .98, "irregular")
    report("(2.20)+(2.22) expected discounted entropy weight", "PASS" if abs(got - expected) < 1e-12 else "FAIL",
           f"code={got:.6f} paper={expected:.6f}")


def check_hold_sampler_empirical():
    """Draw holds from the pinned CT-RL wrapper itself and compare with (4.31)."""
    from bench.env import make_common_env
    env = make_common_env("cheetah-run", seed=3)
    counts = np.zeros(16)
    draws = 0
    for episode in range(10):
        env.reset(seed=3 + episode)
        grid = np.diff(env._time_points)[:-1]   # drop the adjusted final hold
        for h in grid:
            k = int(round(h / .002))
            if 1 <= k <= 15:
                counts[k] += 1
                draws += 1
    if draws < 1000:
        report("(4.31) empirical sampler", "SKIP", "wrapper exposes no standalone hold draw")
        return
    freq = counts[1:] / draws
    report("(4.31) empirical sampler (pinned wrapper)", "PASS" if abs(freq[0] - .891) < .02 and abs(freq[-1] - .099) < .02 else "DIFF",
           f"n={draws} P1={freq[0]:.3f} P15={freq[-1]:.3f} Pmid~{freq[1:-1].mean():.5f}")


# ------------------------------------------------- Prop 2.1 frozen Euler update
def check_euler_contraction():
    from bench.operators import finite_step
    rng = np.random.default_rng(0)
    N = 40
    P = rng.random((N, N)); P /= P.sum(1, keepdims=True)
    gamma = np.full(N, .97)
    b = rng.normal(size=N)
    z, zt = rng.normal(size=N), rng.normal(size=N)
    eta = .5
    out, _ = finite_step(z, b, gamma, P, "ct_picard", eta=eta)
    outt, _ = finite_step(zt, b, gamma, P, "ct_picard", eta=eta)
    factor = np.max(abs(out - outt)) / np.max(abs(z - zt))
    bound = 1 - eta * (1 - gamma.max())
    report("Prop 2.1 Lipschitz factor of (2.19)", "PASS" if factor <= bound + 1e-12 else "FAIL",
           f"measured={factor:.4f} <= bound={bound:.4f}")
    exact, _ = finite_step(z, b, gamma, P, "collocation_direct")
    v = z.copy()
    for _ in range(3000):
        v, _ = finite_step(v, b, gamma, P, "ct_picard", eta=eta)
    report("Prop 2.1 convergence to unique fixed point", "PASS" if np.max(abs(v - exact)) < 1e-8 else "FAIL",
           f"|v_3000 - v*|_inf={np.max(abs(v - exact)):.2e}")
    # (2.19): v+ = (1-w)v + w T v
    T = b + gamma * (P @ z)
    report("(2.19) ct_picard == (1-w)v + w T v", "PASS" if np.allclose(out, (1 - eta) * z + eta * T) else "FAIL", "")


# -------------------------------------------------- Appendix F.1 phase solver
def check_phase_solver():
    from bench.operators import phase_coefficients
    sigma, d, omega = .5, 1., 3.
    c, s = phase_coefficients(sigma, d, omega)
    K_code = np.array([[-d, omega], [-omega, -d]])  # isotropic damping, frequency omega
    Mcode = np.linalg.solve(np.eye(2) - sigma / 2 * K_code, np.eye(2) + sigma / 2 * K_code)
    K_paper = np.array([[0., 1.], [-1., -d]])        # F.1: e' = rho, rho' = -e - d rho
    Mpaper = np.linalg.solve(np.eye(2) - sigma / 2 * K_paper, np.eye(2) + sigma / 2 * K_paper)
    same_code = np.allclose(Mcode, [[c, s], [-s, c]])
    report("App. F.1 phase step = implicit midpoint", "PASS" if same_code else "FAIL",
           "code step is the Cayley map of e'=-d e+w rho, rho'=-w e-d rho")
    report("App. F.1 oscillator matches paper's (e'=rho, rho'=-e-d rho)",
           "PASS" if np.allclose(Mcode, Mpaper) else "DIFF",
           f"code damps both e and rho with frequency w={omega}; paper damps rho only at unit frequency. "
           f"one-step residual contraction code={np.linalg.norm(Mcode, 2):.3f} paper={np.linalg.norm(Mpaper, 2):.3f}")


# ---------------------------------------------------- (4.28) exact OU step
def check_ou():
    from fjbench.environment import ou_step
    rng = np.random.default_rng(1)
    k, s, dt, w0 = 1., 2., .002, .7
    x = np.array([ou_step(w0, k, s, dt, rng) for _ in range(200000)])
    m, v = math.exp(-k * dt) * w0, s * s * (1 - math.exp(-2 * k * dt)) / (2 * k)
    ok = abs(x.mean() - m) < 4 * math.sqrt(v / len(x)) and abs(x.var() / v - 1) < .02
    report("(4.28) exact OU marginal step", "PASS" if ok else "FAIL",
           f"mean {x.mean():.5f}/{m:.5f}, var {x.var():.3e}/{v:.3e}")


# --------------------------------- Appendix A / (2.31): mass matrix and bias
def check_mechanics():
    import mujoco
    import torch
    from dm_control.suite import cheetah
    from pure_cpht.mechanics import Robot

    class Nominal:
        pass
    physics = cheetah.Physics.from_xml_string(*cheetah.get_model_and_assets())
    m, d = physics.model.ptr, physics.data.ptr
    nominal = Nominal(); nominal.model = m; nominal.xml_sha256 = "n/a"
    from pure_cpht.mechanics import public_geometry
    robot = Robot(public_geometry(nominal), dtype=torch.float64)
    rng = np.random.default_rng(2)
    worst_M = worst_c = 0.
    for _ in range(25):
        d.qpos[:] = 0; d.qpos[1] = rng.uniform(-.1, .3); d.qpos[2] = rng.uniform(-.5, .5)
        lo, hi = m.jnt_range[3:, 0], m.jnt_range[3:, 1]
        d.qpos[3:] = rng.uniform(lo, hi)
        d.qvel[:] = rng.normal(0, 2, 9)
        mujoco.mj_forward(m, d)
        from physical_fj.mechanics import full_mass; M = full_mass(m, d)
        q = torch.tensor(d.qpos[None].copy()); v = torch.tensor(d.qvel[None].copy())
        out = robot.quantities(q, v)
        worst_M = max(worst_M, np.max(abs(out["M"][0].numpy() - M)) / np.max(abs(M)))
        # code c = bias (gravity+Coriolis) + spring + damping = qfrc_bias - qfrc_passive
        worst_c = max(worst_c, np.max(abs(out["c"][0].numpy() - (d.qfrc_bias - d.qfrc_passive))) /
                      max(1., np.max(abs(d.qfrc_bias))))
    report("App. A mass matrix M(q) vs MuJoCo mj_fullM", "PASS" if worst_M < 1e-8 else "FAIL", f"max rel err={worst_M:.2e}")
    report("(2.31) bias c - passive vs MuJoCo", "PASS" if worst_c < 1e-8 else "FAIL", f"max rel err={worst_c:.2e}")
    gears = m.actuator_gear[:, 0]
    B = robot.B.numpy()
    report("(2.28) gear diag(120,90,60,90,60,30), rank B = 6 < 9",
           "PASS" if np.allclose(gears, [120, 90, 60, 90, 60, 30]) and np.linalg.matrix_rank(B) == 6 else "FAIL",
           f"gears={gears.tolist()}")


# ----------------------------------- (2.33) implicit-damping Euler substep
def check_euler_substep():
    import mujoco
    from dm_control.suite import cheetah
    physics = cheetah.Physics.from_xml_string(*cheetah.get_model_and_assets())
    m, d = physics.model.ptr, physics.data.ptr
    m.opt.timestep = .002
    rng = np.random.default_rng(4)
    worst = 0.
    for _ in range(20):
        mujoco.mj_resetData(m, d)
        d.qpos[1] = rng.uniform(-.2, 0.); d.qpos[3:] = rng.uniform(m.jnt_range[3:, 0], m.jnt_range[3:, 1])
        d.qvel[:] = rng.normal(0, 1, 9); d.ctrl[:] = rng.uniform(-1, 1, 6)
        mujoco.mj_forward(m, d)
        from physical_fj.mechanics import full_mass; M = full_mass(m, d)
        Ds = np.diag(m.dof_damping)
        dt = m.opt.timestep
        abar = np.linalg.solve(M + dt * Ds, M @ d.qacc)  # (M + dt Ds)^-1 (tau0 + Jc^T f)
        v_pred = d.qvel + dt * abar
        q_pred = d.qpos + dt * v_pred
        mujoco.mj_step(m, d)
        worst = max(worst, np.max(abs(d.qvel - v_pred)), np.max(abs(d.qpos - q_pred)))
    report("(2.33) MuJoCo Euler substep with implicit damping", "PASS" if worst < 1e-9 else "FAIL",
           f"max |pred-actual|={worst:.2e} (20 random contact states)")


# ----------------------------------------- (2.34) reward quadrature convention
def check_reward_quadrature():
    beta = -math.log(.98) / .01        # 1/s
    delta, N = .002, 15
    r = np.random.default_rng(5).random(N)
    j = np.arange(N)
    code = np.sum(np.exp(-beta * j * delta) * (-np.expm1(-beta * delta) / beta) * r)   # bench/env.py (seconds)
    paper = np.sum(delta * np.exp(-beta * (j + 1) * delta) * r)                        # (2.34)
    rel = abs(code - paper) / paper
    report("(2.34) discounted reward quadrature", "DIFF" if rel > 1e-12 else "PASS",
           f"code integrates exp(-beta u) exactly per substep with exponent j*delta (j>=0); paper uses delta*exp(-beta j delta), j>=1. "
           f"relative gap {rel:.2e} for a 15-substep hold (O(beta*delta))")


# ------------------------------------------------------- Table 1 settings
def check_table1():
    """Table 1's F-native/F-position/J-static/J-wind settings in fjbench.config.

    run_experiments.sh uses the terrain-course F/J cases (experiment="contact"),
    so these settings document the remaining gap for a Table 1 replication only.
    """
    from fjbench.config import make_config, observation_dim
    J = make_config("J", "static")
    W = make_config("J", "wind")
    F = make_config("F", "position")
    issues = []
    issues.append(f"J-static friction train={J['friction_train']} (paper Unif[0.6,1.0]); "
                  f"load train={J['load_train']} (paper Unif[0,2] kg); speed g train={J['speed_train']} (paper Unif[6,10])")
    issues.append(f"J-wind kappa={W['wind_kappa']} sigma={W['wind_sigma']} force_scale={W['wind_force_scale']} -> "
                  f"force sigma={W['wind_sigma'] * W['wind_force_scale']:.2f} N/sqrt(s) (paper kappa=1, sigma=2)")
    issues.append(f"J-wind observation dim={observation_dim(W)} (paper: 18, includes o^w = w + eps)")
    issues.append(f"F-position std={F['position_std']} for all 8 channels (paper: 0.005 m root height, 0.01 rad angles)")
    issues.append(f"transfer tests friction={J['friction_test']} load={J['load_test']} speed={J['speed_test']} "
                  f"wind test kappa={W['wind_test_kappa']} sigma={W['wind_test_sigma']} "
                  f"(paper: friction [0.3,0.5], payload [3,4] kg, speed 12, dw=-2w dt+4dW)")
    for i in issues:
        report("Table 1 variant settings (not run by run_experiments.sh)", "DIFF", i)


def check_wind_point():
    from bench.env import make_common_env
    env = make_common_env("cheetah-run", seed=0)
    report("(A.22) J-wind force point (not run by run_experiments.sh)", "PASS" if env.force_body_name == "torso" else "DIFF",
           f"code applies the point force at body '{env.force_body_name}' + 5 cm offset; paper: torso centre of mass")


def check_payload():
    import mujoco
    from dm_control.suite import cheetah
    physics = cheetah.Physics.from_xml_string(*cheetah.get_model_and_assets())
    m = physics.model.ptr
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    mass, I = float(m.body_mass[t]), m.body_inertia[t].copy()
    load = 2.
    code = I * (mass + load) / mass
    paper = I + .4 * load * .05 ** 2  # sphere r=5 cm at the torso COM; (A.10) note "2/5 m_load (0.05)^2"
    report("(A.10) J-static payload inertia (not run by run_experiments.sh)", "DIFF",
           f"code scales torso inertia by (m+load)/m -> Iyy {code[1]:.4f}; paper adds 2/5 m r^2 -> Iyy {paper[1]:.4f} (load 2 kg)")


def check_hold_disclosure():
    """Run the pipeline environment on the irregular clock and read what a policy sees."""
    import numpy as np
    from cpht_compare import config
    from physical_fj.mechanics import PhysicalEnvironment
    profile = config.load(str(ROOT / "experiments" / "smoke.json"))
    job = next(config.jobs(profile))
    s = config.settings_for(profile, job)
    c = config.control_for(profile, s)
    env = PhysicalEnvironment(c, s, 7, clock="irregular", purpose="evaluation")
    seen, requested = set(), set()
    try:
        for _ in range(40):
            seen.add(round(float(env.current["raw"][30]) * c["reference_time"], 6))
            requested.add(round(env.hold(), 6))
            env.step(np.zeros(6, np.float32))
    finally:
        env.close()
    hidden = seen == {c["fixed_hold"]} and len(requested) > 1
    report("Sec 2.6/4.7 unannounced holds", "PASS" if hidden and not s["hold_disclosure"] else "DIFF",
           f"policy input shows hold(s) {sorted(seen)} s while the realized holds were {sorted(requested)} s")


def check_energy_shaping():
    import subprocess
    hits = subprocess.run(["grep", "-rlE", "energy.shap|Dgear|desired_potential", "--include=*.py", "."],
                          capture_output=True, text=True, cwd=ROOT).stdout.split()
    hits = [h for h in hits if "tools/" not in h]
    hits = [h for h in hits if "tests" not in h]
    report("(4.26) energy-shaping structured controller", "DIFF" if not hits else "PASS",
           "no implementation of u=-grad U^a - K v, a=Dgear^-1 u found; pure_cpht/cpht_compare use a periodic "
           "reference + bounded inverse-dynamics QP instead" if not hits else str(hits))


import os
os.chdir(ROOT)
for f in (check_hold_law, check_hold_sampler_empirical, check_euler_contraction, check_phase_solver, check_ou,
          check_mechanics, check_euler_substep, check_reward_quadrature, check_table1, check_wind_point,
          check_payload, check_hold_disclosure, check_energy_shaping):
    try:
        f()
    except Exception as exc:  # keep running other checks
        report(f.__name__, "ERROR", repr(exc)[:300])

summary = {s: sum(1 for _, x, _ in results if x == s) for s in ("PASS", "DIFF", "FAIL", "ERROR", "SKIP")}
print("\nSUMMARY", summary)
sys.exit(1 if summary["FAIL"] or summary["ERROR"] else 0)
