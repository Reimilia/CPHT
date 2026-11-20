"""Tensor-only, held-action pH/contact imagination shared by every comparison.

Contact selection, Coulomb projection and box feedback are piecewise smooth.
The observation approximation transports the current local scan; it acquires no
future measurements and keeps the current material belief fixed over a rollout.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from pure_cpht.contact import terrain

@dataclass
class Imagined:
    objective: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    continuation: torch.Tensor
    raw: torch.Tensor
    phase: torch.Tensor
    survival: torch.Tensor
    actions: torch.Tensor
    q: torch.Tensor
    v: torch.Tensor
    invalid: torch.Tensor
    solver_residual: torch.Tensor
    extrapolation: torch.Tensor
    physical_durations: torch.Tensor

def snapshot(packet, phase, target, hold, correction=None, points=None):
    """Copy only permitted causal inputs, never audit truth or a map object."""
    raw = np.asarray(packet["raw"], dtype=np.float32).copy()
    scan = np.asarray(packet["scan"], dtype=np.float32).copy()
    if raw.shape not in ((31,), (60,)) or scan.shape != (7, 3):
        raise ValueError("Use a versioned 31/60-channel sensing contract")
    estimated = packet.get("environment", {})
    mu = float(estimated.get("friction", raw[46] if len(raw) == 60 else 0.6))
    if correction is None:
        if points is None:
            raise ValueError("Number of public contact points is required")
        correction = np.zeros((points, 2), dtype=np.float32)
    return dict(raw=raw, scan=scan, phase=float(phase), target=float(target),
                first_hold=float(hold), mu=mu,
                correction=np.asarray(correction, dtype=np.float32).copy())

def stack_snapshots(rows, robot):
    return {key: robot.t(np.stack([row[key] for row in rows])) for key in rows[0]}

class Imagination:
    def __init__(self, controller, task, optimization, horizon):
        self.controller = controller
        self.robot = controller.robot
        self.contact = controller.contact
        self.task = dict(task)
        self.options = dict(optimization)
        self.steps = max(1, round(horizon / controller.options["planning_dt"]))
        self.beta = -math.log(task["gamma_reference"]) / task["reference_time"]

    def expand(self, context, count):
        return {k: value.expand((count,) + value.shape[1:]) if len(value) == 1 else value
                for k, value in context.items()}

    def observation(self, q, v, port, context, elapsed, quantities, goal):
        r = self.robot
        original = context["raw"]
        remaining = (goal - quantities["com_position"][:, 0]) / self.task["path_length"]
        left = (original[:, 28] - elapsed / self.task["horizon"]).clamp_min(0)
        task = torch.stack((remaining, left, context["target"] / 10,
                            torch.full_like(left, self.controller.options["planning_dt"] /
                                            self.task["reference_time"])), -1)
        raw = torch.cat((q, v, port, task), -1)
        if original.shape[-1] == 60:
            offsets = r.t(self.task["contact_offsets"])
            root_x = q[:, 0] + r.positions[1, 0]
            root_z = q[:, 1] + r.positions[1, 1]
            h, slope = terrain(context["scan"], root_x[:, None] + offsets[None])
            extra = original[:, 31:]
            # Preserve current covariance/validity; ages advance. No invented evidence.
            raw = torch.cat((raw, h - root_z[:, None], slope, extra[:, 14:20],
                             (extra[:, 20:22] + elapsed[:, None]).clamp_max(5),
                             extra[:, 22:26],
                             (extra[:, 26:27] + elapsed[:, None]).clamp_max(5),
                             extra[:, 27:29]), -1)
        return raw

    def advance(self, q, v, action, context, duration, elapsed):
        """One held command, with current inferred contact recomputed each microstep."""
        r = self.robot
        micro = self.controller.options["planning_dt"] / self.controller.options["substeps"]
        count = max(1, math.ceil(float(duration.detach().max()) / micro - 1e-8))
        dt = duration / count
        reward = torch.zeros(len(q), device=q.device, dtype=q.dtype)
        discount = torch.ones_like(reward)
        solver = torch.zeros_like(reward)
        invalid = torch.zeros(len(q), device=q.device, dtype=torch.bool)
        d = r.quantities(q, v)
        port = context["raw"][:, 18:27]
        c_beta = -torch.expm1(-self.beta * dt) / self.beta
        extrapolation = torch.zeros_like(reward)
        for substep in range(count):
            # elapsed is a vector for variable recorded holds in replay batches.
            # Contact.observe residual corrections decay causally from the start.
            correction = context["correction"] * torch.exp(
                -elapsed / self.controller.options["observer_decay"])[:, None, None]
            contact = self.contact.solve(q, v, action, context["scan"], d,
                                         correction, elapsed=0.0,
                                         friction=context["mu"][:, None])
            port = contact["port"]
            acceleration = torch.linalg.solve(
                d["M"], (action @ r.B.T + port - d["c"])[:, :, None]).squeeze(-1)
            v = v + dt[:, None] * acceleration
            q = q + dt[:, None] * v
            invalid = invalid | (~torch.isfinite(q).all(-1)) | (~torch.isfinite(v).all(-1))
            invalid = invalid | (v.abs().max(-1).values > 80)
            q = torch.nan_to_num(q).clamp(-100, 100)
            v = torch.nan_to_num(v).clamp(-100, 100)
            d = r.quantities(q, v)
            relative = (d["com_velocity"][:, 0] - context["target"]) / context["target"]
            rate = -(self.task["path_speed_weight"] * relative.square()
                     + self.task["path_effort_weight"] * action.square().mean(-1)
                     + self.task["path_time_weight"]
                     + self.task["path_backwards_weight"] *
                     (-d["com_velocity"][:, 0] / context["target"]).clamp_min(0))
            reward = reward + discount * c_beta * rate
            discount = discount * torch.exp(-self.beta * dt)
            solver = torch.maximum(solver, contact["relative_projected_residual"])
            outside = ((d["points"][:, :, 0] < context["scan"][:, :1, 0]) |
                       (d["points"][:, :, 0] > context["scan"][:, -1:, 0])).to(q.dtype)
            extrapolation = torch.maximum(extrapolation, outside.mean(-1))
            elapsed = elapsed + dt
            self.controller.predicted_steps += len(q)
        return q, v, port, d, reward, discount, invalid, solver, extrapolation, elapsed

    def rollout(self, context, parameters=None, selector=None, value=None):
        if (parameters is None) == (selector is None):
            raise ValueError("Supply either fixed motion parameters or a selector")
        count = len(parameters) if parameters is not None else len(context["raw"])
        context = self.expand(context, count)
        if any(len(x) != count for x in context.values()):
            raise ValueError("Mismatched rollout batch")
        r = self.robot
        q, v = context["raw"][:, :9], context["raw"][:, 9:18]
        phase = context["phase"]
        port = context["raw"][:, 18:27]
        elapsed = torch.zeros(count, device=q.device, dtype=q.dtype)
        survival = torch.ones_like(elapsed)
        global_discount = torch.ones_like(elapsed)
        score = torch.zeros_like(elapsed)
        initial = r.quantities(q, v)
        goal = initial["com_position"][:, 0] + context["raw"][:, 27] * self.task["path_length"]
        raw = context["raw"]
        raws, phases, survivals = [raw], [phase], [survival]
        rewards, discounts, continues, actions, durations = [], [], [], [], []
        invalid = torch.zeros(count, device=q.device, dtype=torch.bool)
        residual = torch.zeros_like(elapsed)
        extrapolation = torch.zeros_like(elapsed)
        next_plan = torch.zeros_like(elapsed)
        motion = parameters
        for j in range(self.steps):
            if selector is not None:
                proposal = selector(raw, phase)
                due = elapsed >= next_plan - 1e-8
                motion = proposal if motion is None else torch.where(due[:, None], proposal, motion)
                next_plan = torch.where(due, elapsed + self.controller.options["replan_seconds"], next_plan)
            action, qp_residual = self.controller.feedback(
                q, v, port, context["scan"], motion, phase, context["target"])
            duration = (context["first_hold"] if j == 0 else
                        torch.full_like(elapsed, self.controller.options["planning_dt"]))
            q, v, port, d, payment, gamma, bad, contact_residual, outside, elapsed = self.advance(
                q, v, action, context, duration, elapsed)
            phase = phase + 2 * math.pi * motion[:, 18] * duration
            root_x = q[:, 0] + r.positions[1, 0]
            root_z = q[:, 1] + r.positions[1, 1]
            height, _ = terrain(context["scan"], root_x[:, None])
            clearance = root_z - height[:, 0]
            softness = self.options["failure_softness"]
            fall_height = torch.sigmoid((self.task["path_fall_clearance"] - clearance) / softness)
            fall_pitch = torch.sigmoid((q[:, 2].abs() - self.task["path_fall_pitch"]) / softness)
            fall = 1 - (1 - fall_height) * (1 - fall_pitch)
            goal_prob = torch.sigmoid((d["com_position"][:, 0] - goal) / 0.05)
            remaining = (context["raw"][:, 28] * self.task["horizon"] - elapsed).clamp_min(0)
            timeout = (remaining <= 0).to(q.dtype)
            stall = self.task["path_speed_weight"] + self.task["path_time_weight"]
            absorbing_tail = stall * (-torch.expm1(-self.beta * remaining)) / self.beta
            event = (-fall * (self.task["path_failure_penalty"] + absorbing_tail)
                     + (1 - fall) * goal_prob * self.task["path_goal_bonus"]
                     - (1 - fall) * (1 - goal_prob) * timeout * self.task["path_failure_penalty"])
            payment = payment + gamma * event - bad.to(q.dtype) * 1000
            continuation = (1 - fall) * (1 - goal_prob) * (1 - timeout)
            score = score + global_discount * survival * payment
            survival = survival * continuation
            global_discount = global_discount * gamma
            raw = self.observation(q, v, port, context, elapsed, d, goal)
            invalid = invalid | bad
            residual = torch.maximum(residual, torch.maximum(qp_residual, contact_residual))
            extrapolation = torch.maximum(extrapolation, outside)
            rewards.append(payment)
            discounts.append(gamma)
            continues.append(continuation)
            actions.append(action)
            durations.append(duration)
            raws.append(raw)
            phases.append(phase)
            survivals.append(survival)
        if value is not None:
            score = score + global_discount * survival * value(raw, phase)
        return Imagined(score, torch.stack(rewards), torch.stack(discounts),
                        torch.stack(continues), torch.stack(raws), torch.stack(phases),
                        torch.stack(survivals), torch.stack(actions), q, v, invalid,
                        residual, extrapolation, torch.stack(durations))

