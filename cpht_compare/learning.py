"""BPTT selector learning and variable-hold TD(lambda) task-value fitting."""
from collections import deque
import numpy as np
import torch
from .networks import lambda_targets
from .rollout import snapshot, stack_snapshots

class Replay:
    def __init__(self, capacity, seed, rows=()):
        self.rows = deque(rows, maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, row):
        self.rows.append({key: np.asarray(value).copy() for key, value in row.items()})

    def sample(self, count):
        ids = self.rng.integers(0, len(self.rows), size=count)
        return [self.rows[int(i)] for i in ids]

    def state(self):
        return dict(rows=list(self.rows), rng=self.rng.bit_generator.state)

    def restore(self, state):
        self.rows = deque(state["rows"], maxlen=self.rows.maxlen)
        self.rng.bit_generator.state = state["rng"]

def calibration_replay(data, normalization, points, capacity, seed, hold=None):
    """Seed imagination with calibration states.

    hold=None uses each record's requested hold (disclosed-hold interface);
    a number imposes that nominal first hold (unannounced holds).
    """
    replay = Replay(capacity, seed)
    rng = np.random.default_rng(seed + 1)
    for i in normalization["train_indices"]:
        row = snapshot(dict(raw=data["raw"][i], scan=data["scan"][i]),
                       rng.uniform(-np.pi, np.pi), float(data["raw"][i, 29] * 10),
                       float(data["h_requested"][i]) if hold is None else float(hold), points=points)
        replay.add(row)
    return replay

class Trainer:
    def __init__(self, controller, options):
        self.selector = controller.selector
        self.value = controller.value
        self.options = dict(options)
        self.optimizer = torch.optim.Adam(self.selector.parameters(), lr=options["selector_lr"])
        self.value_optimizer = (torch.optim.Adam(self.value.parameters(), lr=options["selector_lr"])
                                if self.value is not None else None)
        self.updates = 0
        self.value_updates = 0
        self.skipped = 0

    def update(self, controller, replay):
        options = self.options
        context = stack_snapshots(replay.sample(options["batch_size"]), controller.robot)
        self.selector.train()
        if self.value is not None:
            self.value.requires_grad_(False)
        result = controller.engine.rollout(context, selector=self.selector, value=self.value)
        duration = result.physical_durations.sum(0).mean().detach().clamp_min(0.01)
        loss = -result.objective.mean() / duration
        self.optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            if self.value is not None:
                self.value.requires_grad_(True)
            self.skipped += 1
            return dict(skipped=True, reason="nonfinite_imagined_objective")
        loss.backward()
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                     for p in self.selector.parameters())
        if not finite:
            if self.value is not None:
                self.value.requires_grad_(True)
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            return dict(skipped=True, reason="nonfinite_selector_gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(self.selector.parameters(), options["grad_clip"]))
        self.optimizer.step()
        self.updates += 1
        metrics = dict(skipped=False, selector_loss=float(loss.detach()),
                       predicted_return=float(result.objective.detach().mean()),
                       gradient_norm=norm, selector_updates=self.updates,
                       imagined_invalid_fraction=float(result.invalid.float().mean()),
                       imagined_solver_residual=float(result.solver_residual.detach().max()),
                       terminal_survival=float(result.survival[-1].detach().mean()))
        if self.value is not None:
            self.value.requires_grad_(True)
            raw, phase = result.raw.detach(), result.phase.detach()
            shape = raw.shape[:2]
            flat = raw.reshape(-1, raw.shape[-1])
            with torch.no_grad():
                values = self.value(flat, phase.reshape(-1)).reshape(shape)
                targets = lambda_targets(result.rewards.detach(), result.discounts.detach(),
                                         result.continuation.detach(), values, options["gae_lambda"])
            x = raw[:-1].reshape(-1, raw.shape[-1])
            angle = phase[:-1].reshape(-1)
            target = targets.reshape(-1)
            for _ in range(options["critic_epochs"]):
                predicted = self.value(x, angle)
                value_loss = (predicted - target).square().mean()
                if not bool(torch.isfinite(value_loss)):
                    metrics["critic_skipped"] = "nonfinite_critic_loss"
                    break
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
                           for p in self.value.parameters()):
                    self.value_optimizer.zero_grad(set_to_none=True)
                    metrics["critic_skipped"] = "nonfinite_critic_gradient"
                    break
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), options["grad_clip"])
                self.value_optimizer.step()
                self.value_updates += 1
            metrics.update(critic_loss=float(value_loss.detach()), critic_updates=self.value_updates)
        return metrics

    def state(self):
        return dict(optimizer=self.optimizer.state_dict(),
                    value_optimizer=self.value_optimizer.state_dict() if self.value_optimizer else None,
                    updates=self.updates, value_updates=self.value_updates, skipped=self.skipped)

    def restore(self, saved):
        self.optimizer.load_state_dict(saved["optimizer"])
        if self.value_optimizer:
            self.value_optimizer.load_state_dict(saved["value_optimizer"])
        self.updates, self.value_updates, self.skipped = saved["updates"], saved["value_updates"], saved["skipped"]

