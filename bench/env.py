"""DMC environments with explicit physical time and off-centre point forces."""
from __future__ import annotations

import math
import importlib
import sys
import types
import numpy as np
from .protocol import source_root, env_kwargs, TASKS, CONTEXTS


def entropy_weight(task, native_dt, gamma_native, clock):
    """E[integral_0^(h/native_dt) exp(-beta*t) dt] for the renewal clock.

    The final clipped interval can differ. Its frequency is logged through dt.
    """
    t = TASKS[task]
    beta = -math.log(gamma_native)
    if clock == "regular":
        intervals, probabilities = np.array([t["dt"]]), np.array([1.])
    else:
        nlo = round(t["min_dt"] / t["physics_dt"])
        nhi = round(t["max_dt"] / t["physics_dt"])
        intervals = np.arange(nlo, nhi + 1) * t["physics_dt"]
        probabilities = np.zeros(len(intervals))
        probabilities[0] = t["tail_p"] * t["tail_split"]
        probabilities[-1] = t["tail_p"] * (1 - t["tail_split"])
        probabilities[1:-1] = (1 - t["tail_p"]) / (len(intervals) - 2)
    return float(probabilities @ (-np.expm1(-beta * intervals / native_dt) / beta))


def make_paper_env(task, seed, clock="irregular"):
    source_root()
    from environment.dmc import DMCContinuousEnv
    domain, name = task.split("-", 1)
    return DMCContinuousEnv(domain, name, seed=seed, **env_kwargs(task, clock))


def make_common_env(task, seed, clock="irregular", context="nominal", gamma_native=.98):
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context {context}")
    root=source_root()
    import mujoco
    from dm_control import rl
    # Load the hash-checked upstream environment modules in a private package.
    # Upstream environment/__init__.py eagerly imports its Torch logger, although
    # base.py and dmc.py only need the simulator. Previews need no training stack.
    namespace="bench._ct_environment"
    if namespace not in sys.modules:
        package=types.ModuleType(namespace)
        package.__path__=[str(root/"environment")]
        package.__package__=namespace
        sys.modules[namespace]=package
    module=importlib.import_module(namespace+".dmc")
    DMCContinuousEnv,_flatten_obs=module.DMCContinuousEnv,module._flatten_obs

    class CommonDMC(DMCContinuousEnv):
        def __init__(self):
            domain, name = task.split("-", 1)
            super().__init__(domain, name, seed=seed, **env_kwargs(task, clock))
            # The outer wrapper counts decisions. Inner calls count physics substeps.
            self._env._step_limit = float("inf")
            self.context = context
            self.gamma_native = gamma_native
            self._beta_native = -math.log(gamma_native)
            physics = self._env.physics
            model, data = physics.model, physics.data
            if context == "damping_1p5":
                model.dof_damping[:] *= 1.5
            if context == "mass_1p2":
                model.body_mass[1:] *= 1.2
                model.body_inertia[1:] *= 1.2
                mujoco.mj_setConst(model.ptr, data.ptr)
            depths = np.zeros(model.nbody, dtype=int)
            for i in range(1, model.nbody):
                depths[i] = depths[model.body_parentid[i]] + 1
            candidates = [i for i in range(1, model.nbody) if model.body_mass[i] > 0]
            self.force_body = max(candidates, key=lambda i: (depths[i], model.body_mass[i], -i))
            self.force_body_name = model.id2name(self.force_body, "body")
            # Offset the application point from this body's centre of mass by 5 cm.
            self.force_point_local = model.body_ipos[self.force_body].copy() + np.array([.05, 0., 0.])
            self.total_mass = float(np.sum(model.body_mass))
            self._jacp = np.zeros((3, model.nv))
            self._jacr = np.zeros((3, model.nv))

        def _point_force(self):
            p = self._env.physics
            p.data.qfrc_applied[:] = 0.
            p.data.xfrc_applied[:] = 0.
            time = self.cur_t + self._micro_elapsed
            active = context == "point_drag" or (
                context == "point_pulse" and .25 * self.episode_duration <= time < .40 * self.episode_duration)
            if not active:
                return 0., 0., 0
            body = self.force_body
            point = p.data.xpos[body] + p.data.xmat[body].reshape(3, 3) @ self.force_point_local
            mujoco.mj_jac(p.model.ptr, p.data.ptr, self._jacp, self._jacr, point, body)
            point_velocity = self._jacp @ p.data.qvel
            if context == "point_drag":
                force = .05 * self.total_mass * (np.array([1., 0., 0.]) - point_velocity)
            else:
                force = np.array([.05 * self.total_mass * abs(p.model.opt.gravity[2]), 0., 0.])
            p.data.qfrc_applied[:] = self._jacp.T @ force
            power = float(force @ point_velocity)
            active_dofs = int(np.count_nonzero(np.abs(p.data.qfrc_applied) > 1e-10))
            return power, float(np.linalg.norm(p.data.qfrc_applied)), active_dofs

        def _step_physics(self, action, dt):
            pd = self._get_physics_dt()
            count = max(1, int(round(dt / pd)))
            restore_dt, restore_nsub, _ = self._set_control_dt_for_one_step(pd)
            integral, discounted, work = 0., 0., 0.
            force_norm, active_dofs = 0., 0
            term, trunc = False, False
            ts = None
            weight = -math.expm1(-self._beta_native * pd / self.dt_default) / self._beta_native
            completed = 0
            self._micro_elapsed = 0.
            try:
                for j in range(count):
                    self._micro_elapsed = j * pd
                    power, norm, dofs = self._point_force()
                    ts = self._env.step(np.asarray(action, dtype=np.float32))
                    if hasattr(self, "_after_physics_step"):
                        self._after_physics_step(pd)
                    completed += 1
                    rate = float(ts.reward or 0.)
                    integral += rate * pd
                    discounted += math.exp(-self._beta_native * j * pd / self.dt_default) * weight * rate
                    work += power * pd
                    force_norm = max(force_norm, norm)
                    active_dofs = max(active_dofs, dofs)
                    if ts.last():
                        term = ts.discount is not None and float(ts.discount) == 0.
                        trunc = not term
                        break
            except rl.control.PhysicsError as exc:
                # Failed seeds must remain visible. Never turn them into successful zero-return runs.
                raise RuntimeError(f"MuJoCo failure in {task}, seed={seed}, context={context}") from exc
            finally:
                self._restore_control_dt(restore_dt, restore_nsub)
            obs = _flatten_obs(ts.observation)
            if not np.isfinite(obs).all() or not np.isfinite(discounted):
                raise FloatingPointError("Non-finite physics observation or reward")
            self._last_obs_dmc = obs
            actual = completed * pd
            self._elapsed_time += actual
            info = dict(dt_requested=float(dt), dt_used=actual, physics_steps=completed,
                        endpoint_reward_rate=float(ts.reward or 0.), reward_time_integral=integral,
                        discounted_reward_integral=discounted, external_work=work,
                        max_generalized_force=force_norm, directly_forced_dofs=active_dofs,
                        force_body=self.force_body_name, context=context)
            return obs, discounted, term, trunc, info, actual

        def step(self, action):
            _, _, _, reward, obs, _, term, trunc, info = self.step_dt(action)
            if self._time_points is not None and self._step_index >= len(self._time_points) - 1:
                trunc = trunc or not term
                info["time_grid_exhausted"] = True
            return obs, reward, term, trunc, info

    return CommonDMC()
