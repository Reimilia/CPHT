"""Shared bounded-feedback deployment with CEM, shooting, or a learned selector."""
import math
import time as wall_time
import numpy as np
import torch
from pure_cpht.controller import Controller
from .networks import Selector, Value
from .rollout import Imagination, snapshot, stack_snapshots

class ComparisonController(Controller):
    def __init__(self, model, options, neutral, task, optimization, horizon,
                 method="cem", normalization=None, seed=0, device="cpu", gait=None):
        super().__init__(model, options, neutral, seed, device, gait)
        self.method = method
        self.optimization = dict(optimization)
        self.engine = Imagination(self, task, optimization, horizon)
        self.selector = None
        self.value = None
        self.context = None
        self._packet = None
        self.last_search = {}
        self.failure_counts = {}
        if method in ("selector", "selector_value"):
            if normalization is None:
                raise ValueError("Training-calibration normalization is required")
            self.selector = Selector(normalization["raw_mean"], normalization["raw_scale"],
                                     self.low.cpu(), self.high.cpu(), self.mean.cpu(),
                                     optimization["selector_hidden"]).to(device)
            if method == "selector_value":
                self.value = Value(self.selector, optimization["selector_hidden"], task,
                                   optimization["critic_scale"]).to(device)

    def failure(self, name):
        self.failure_counts[name] = self.failure_counts.get(name, 0) + 1

    def make_context(self):
        row = snapshot(self._packet, self.phase, self._target, self.first_hold,
                       self.force_correction.detach().cpu().numpy())
        return stack_snapshots([row], self.robot)

    def rollout(self, q0, v0, scan, parameters, target, phase):
        # CEM and shooting use exactly the same differentiable prediction/cost.
        result = self.engine.rollout(self.context, parameters=parameters)
        self.last_search = dict(solver_residual_max=float(result.solver_residual.max().detach()),
                                extrapolation_fraction=float(result.extrapolation.max().detach()),
                                imagined_duration=float(result.physical_durations[:, 0].sum().detach()))
        return -result.objective, result.invalid

    def plan(self, q, v, scan, target):
        self.context = self.make_context()
        started = wall_time.perf_counter()
        if self.method == "cem":
            super().plan(q, v, scan, target)
        elif self.method == "shooting":
            self.shooting()
        else:
            with torch.no_grad():
                self.mean = self.selector(self.context["raw"], self.context["phase"])[0]
            self.plans += 1
            self.last_search = dict(selector_inference=True)
        self.diagnostics.update(self.last_search, planner_seconds=wall_time.perf_counter() - started,
                                method=self.method, failure_counts=dict(self.failure_counts),
                                control_optimizations=self.plans, predictive_substeps=self.predicted_steps)

    def shooting(self):
        with torch.enable_grad():
            parameters = self.mean.detach().clone().unsqueeze(0).requires_grad_()
            optimizer = torch.optim.Adam([parameters], lr=self.optimization["shooting_lr"])
            with torch.no_grad():
                initial = self.engine.rollout(self.context, parameters=parameters)
                initial_score = float(initial.objective[0])
            best_score = initial_score
            best = parameters.detach().clone()
            gradients = []
            iterations = 0
            for _ in range(self.optimization["shooting_steps"]):
                result = self.engine.rollout(self.context, parameters=parameters)
                loss = -result.objective.mean()
                if not bool(torch.isfinite(loss)):
                    self.failure("nonfinite_shooting_objective")
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = parameters.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    self.failure("nonfinite_shooting_gradient")
                    break
                norm = float(torch.linalg.vector_norm(gradient))
                gradients.append(norm)
                if norm > self.optimization["max_grad_norm"]:
                    self.failure("large_shooting_gradient")
                torch.nn.utils.clip_grad_norm_([parameters], self.optimization["grad_clip"])
                optimizer.step()
                with torch.no_grad():
                    parameters.copy_(torch.maximum(self.low, torch.minimum(self.high, parameters)))
                    candidate = self.engine.rollout(self.context, parameters=parameters)
                    score = float(candidate.objective[0])
                    if math.isfinite(score) and not bool(candidate.invalid[0]) and score > best_score:
                        best, best_score = parameters.detach().clone(), score
                iterations += 1
            self.mean = best[0]
            self.plans += 1
            self.last_search = dict(
                predicted_initial_return=initial_score, predicted_optimized_return=best_score,
                predicted_search_gain=best_score - initial_score,
                shooting_iterations=iterations, gradient_norm_max=max(gradients, default=0.0),
                solver_residual_max=float(initial.solver_residual.max()),
                extrapolation_fraction=float(initial.extrapolation.max()),
                imagined_duration=float(initial.physical_durations[:, 0].sum()))

    def act(self, packet, target, hold, time):
        self._packet, self._target = packet, float(target)
        started = wall_time.perf_counter()
        action = super().act(packet, target, hold, time)
        self.context = self.make_context()
        self.diagnostics.update(
            decision_seconds=wall_time.perf_counter() - started,
            method=self.method, failure_counts=dict(self.failure_counts))
        return action

    def current_snapshot(self):
        return {key: value[0].detach().cpu().numpy().copy() for key, value in self.context.items()}

    @torch.no_grad()
    def prediction(self, before, after, action, duration):
        context = self.context
        q, v = context["raw"][:, :9], context["raw"][:, 9:18]
        d = torch.full((1,), duration, device=q.device, dtype=q.dtype)
        q, v, _, _, _, _, invalid, residual, outside, _ = self.engine.advance(
            q, v, self.robot.t(action[None]), context, d, torch.zeros_like(d))
        q_error = q[0] - self.robot.t(after["raw"][:9])
        v_error = v[0] - self.robot.t(after["raw"][9:18])
        # Respect angular branches for this local chart.
        angles = torch.atan2(q_error[2:].sin(), q_error[2:].cos())
        result = dict(
            translation_prediction_l2=float(torch.linalg.vector_norm(q_error[:2])),
            angle_prediction_l2=float(torch.linalg.vector_norm(angles)),
            velocity_prediction_l2=float(torch.linalg.vector_norm(v_error)),
            contact_solver_residual=float(residual[0]),
            scan_extrapolation_fraction=float(outside[0]),
            predicted_invalid=bool(invalid[0]))
        if result["velocity_prediction_l2"] > self.optimization["prediction_error_limit"]:
            self.failure("large_one_step_model_error")
        if result["contact_solver_residual"] > self.optimization["solver_residual_limit"]:
            self.failure("large_contact_solver_residual")
        return result

    def learned_state(self):
        return dict(
            selector=self.selector.state_dict() if self.selector is not None else None,
            value=self.value.state_dict() if self.value is not None else None)

    def load_learned(self, saved):
        if self.selector is not None:
            self.selector.load_state_dict(saved["selector"])
        if self.value is not None:
            self.value.load_state_dict(saved["value"])

