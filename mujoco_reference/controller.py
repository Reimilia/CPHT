"""Exact current-mechanics LS and full-contact MuJoCo shooting controls.

These are privileged references. MuJoCo's contact optimization does not choose
the task objective, acceleration target, gait, or high-level control command.
"""
import math
import time
import numpy as np
import torch
import mujoco
from pure_cpht.controller import Controller
from .common import copied, acceleration_map, bounded_ls, step_native


class NativeController(Controller):
    def __init__(self, env, saved, method, options, seed, fd_epsilon=1e-3, trust_radius=.5):
        super().__init__(saved['pure_model'], options, saved['neutral'], seed,
                         gait=saved['gait'], feedback_only=method != 'mujoco_mpc')
        self.env = env
        physics = env.base._env.physics
        self.model = physics.model.ptr
        self.live = physics.data.ptr
        self.legacy = bool(physics.legacy_step)
        self.method = method
        self.fd_epsilon = fd_epsilon
        self.trust_radius = trust_radius
        self.B = np.vstack([np.zeros((3, 6)), np.diag(self.model.actuator_gear[:, 0])])
        self.native_steps = self.forward_calls = self.feedback_solves = 0
        if (self.model.nq, self.model.nv, self.model.nu, self.model.na) != (9, 9, 6, 0):
            raise ValueError('Native references currently support the 9-coordinate, direct-motor cheetah')
        if self.model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError('The recorded reference integrator must be Euler')
        if not np.allclose(self.model.actuator_ctrlrange, [[-1., 1.]]*6):
            raise ValueError('Expected normalized cheetah actuator bounds')
        # Verify the linear actuation assumption, instead of treating inverse
        # generalized forces as nine independently controllable actuators.
        probe = copied(self.model, self.live)
        probe.ctrl[:] = np.linspace(-.4, .4, 6)
        mujoco.mj_forward(self.model, probe)
        if not np.allclose(probe.qfrc_actuator, self.B@probe.ctrl, atol=1e-10):
            raise ValueError('Actuation differs from the declared linear gear map')

    def forward(self, data, action=None):
        if action is not None:
            data.ctrl[:] = action
        mujoco.mj_forward(self.model, data)
        mujoco.mj_subtreeVel(self.model, data)
        self.forward_calls += 1

    def desired(self, data, scan, parameters, phase, target):
        o = self.options
        qr, vr, ar = self.references(self.robot.t(parameters)[None], phase)
        qr, vr, ar = (x[0].cpu().numpy().astype(float) for x in (qr, vr, ar))
        torso = self.env.base._env.physics.model.name2id('torso', 'body')
        root = data.xpos[torso]
        height = np.interp(root[0], scan[:, 0], scan[:, 1])
        return np.r_[o['speed_kd']*(target-data.subtree_linvel[torso, 0]),
                     o['height_kp']*(o['target_clearance']-(root[2]-height))-o['height_kd']*data.qvel[1],
                     -o['pitch_kp']*data.qpos[2]-o['pitch_kd']*data.qvel[2],
                     ar+o['joint_kp']*(qr-data.qpos[3:])+o['joint_kd']*(vr-data.qvel[3:])]

    def fit(self, data, scan, parameters, phase, target, finite_difference=False, diagnose=False):
        self.forward(data)
        desired = self.desired(data, scan, parameters, phase, target)
        center = data.ctrl.copy()
        base_acceleration = data.qacc.copy()
        jacobian, offset = acceleration_map(self.model, data, self.B)
        low, high = -np.ones(6), np.ones(6)
        if finite_difference:
            # Every perturbation starts with the same warm start/integration
            # state, so differences do not contain a sequential solver history.
            jacobian = np.empty((9, 6))
            for j in range(6):
                minus, plus = center.copy(), center.copy()
                minus[j] = max(-1., center[j]-self.fd_epsilon)
                plus[j] = min(1., center[j]+self.fd_epsilon)
                dm, dp = copied(self.model, data), copied(self.model, data)
                self.forward(dm, minus)
                self.forward(dp, plus)
                jacobian[:, j] = (dp.qacc-dm.qacc)/(plus[j]-minus[j])
            offset = base_acceleration-jacobian@center
            low, high = np.maximum(-1., center-self.trust_radius), np.minimum(1., center+self.trust_radius)
        action, diagnostics = bounded_ls(jacobian, offset, desired, self.options['feedback_ridge'], low, high)
        self.feedback_solves += 1
        if diagnose:
            actual = copied(self.model, data)
            self.forward(actual, action)
            diagnostics.update(acceleration_target=desired, acceleration_linear_prediction=jacobian@action+offset,
                               acceleration_full_contact=actual.qacc.copy(),
                               contact_linearization_error=float(np.linalg.norm(actual.qacc-(jacobian@action+offset))),
                               action_saturation=float(np.mean(abs(action) > .99)),
                               acceleration_jacobian_rank=int(np.linalg.matrix_rank(jacobian)),
                               acceleration_jacobian_singular_values=np.linalg.svd(jacobian, compute_uv=False))
        return action, diagnostics

    def rollout(self, q0, v0, scan, parameters, target, phase):
        # Full current model is read, including terrain/friction outside the
        # causal scan. No future intervention schedule, sensor queue, or clock
        # RNG is accessed. MjModel is never mutated, and every candidate has its
        # own copied MjData. The live environment cannot advance here.
        o = self.options
        scan_np = scan.cpu().numpy().astype(float)
        torso = self.env.base._env.physics.model.name2id('torso', 'body')
        costs, failures = [], []
        for candidate in parameters.cpu().numpy():
            d = copied(self.model, self.live)
            self.forward(d)
            start = d.subtree_com[torso, 0]
            elapsed, angle, cost, invalid = 0., phase, 0., False
            for interval in range(o['horizon_steps']):
                duration = self.first_hold if interval == 0 else o['planning_dt']
                count = max(1, int(round(duration/self.model.opt.timestep)))
                action, _ = self.fit(d, scan_np, candidate, angle, target)
                d.ctrl[:] = action
                # Restore the split-step kinematic invariant after mj_forward.
                mujoco.mj_step1(self.model, d)
                for _ in range(count):
                    step_native(self.model, d, self.legacy)
                    mujoco.mj_subtreeVel(self.model, d)
                    self.native_steps += 1
                    dt = self.model.opt.timestep
                    root = d.xpos[torso]
                    clearance = root[2]-np.interp(root[0], scan_np[:, 0], scan_np[:, 1])
                    speed = d.subtree_linvel[torso, 0]
                    stage = o['speed_weight']*(speed-target)**2+o['clearance_weight']*(clearance-o['target_clearance'])**2
                    stage += o['pitch_weight']*d.qpos[2]**2+o['effort_weight']*float(action@action)
                    # The exact simulator has terrain outside the estimated scan;
                    # retain an explicit scan-coverage penalty for the torso.
                    stage += o['uncertainty_weight']*float(root[0] < scan_np[0, 0] or root[0] > scan_np[-1, 0])
                    cost += dt*stage
                    elapsed += dt
                    angle += 2*math.pi*candidate[-1]*dt
                    invalid = bool(not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                                   or abs(d.qvel).max() > 80 or clearance < .15 or abs(d.qpos[2]) > 1.6
                                   or np.any(d.qpos[3:] < self.model.jnt_range[3:, 0]-.3)
                                   or np.any(d.qpos[3:] > self.model.jnt_range[3:, 1]+.3))
                    if invalid:
                        break
                if invalid:
                    break
            speed = d.subtree_linvel[torso, 0]
            cost += o['progress_weight']*max(0., target*elapsed-(d.subtree_com[torso, 0]-start))+.1*(speed-target)**2
            costs.append(cost+1e6*invalid)
            failures.append(invalid)
        self.predicted_steps = self.native_steps
        return self.robot.t(costs), torch.as_tensor(failures, device=parameters.device)

    @torch.no_grad()
    def act(self, packet, target, hold, time_now):
        started = time.perf_counter()
        if self.last_time is not None:
            self.phase = (self.phase+2*math.pi*float(self.mean[-1])*max(0., time_now-self.last_time))%(2*math.pi)
        self.last_time, self.first_hold = time_now, float(hold)
        if self.method == 'mujoco_mpc' and time_now-self.last_plan >= self.options['replan_seconds']-1e-9:
            self.plan(self.robot.t(packet['raw'][:9])[None], self.robot.t(packet['raw'][9:18])[None],
                      self.robot.t(packet['scan']), target)
            self.last_plan = time_now
        action, diag = self.fit(copied(self.model, self.live), np.asarray(packet['scan']),
                                self.mean.cpu().numpy(), self.phase, target,
                                finite_difference=self.method == 'mujoco_fd_ls', diagnose=True)
        self.previous = self.robot.t(action)
        self.diagnostics.update(diag, native_predictive_steps=self.native_steps,
                                native_forward_calls=self.forward_calls, feedback_solves=self.feedback_solves,
                                control_optimizations=self.plans, phase=self.phase,
                                decision_wall_seconds=time.perf_counter()-started,
                                actor_updates=0, critic_updates=0, model_updates=0)
        # The real environment uses float32 controls; return the exact same
        # command representation that appears in the recorded transition.
        return action.astype(np.float32)
