"""Motion-coefficient selector and optional task-value critic."""
import torch
from torch import nn

def mlp(dimension, widths, output):
    layers = []
    for width in widths:
        layers += [nn.Linear(dimension, width), nn.ELU()]
        dimension = width
    layers.append(nn.Linear(dimension, output))
    return nn.Sequential(*layers)

class Selector(nn.Module):
    def __init__(self, mean, scale, low, high, base, widths):
        super().__init__()
        for name, value in (("mean", mean), ("scale", scale), ("low", low),
                            ("high", high), ("base", base)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        self.net = mlp(len(mean) + 2, widths, 19)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("motion_scale", torch.tensor([0.3] * 6 + [0.5] * 12 + [0.5]))

    def features(self, raw, phase):
        phase = torch.as_tensor(phase, device=raw.device, dtype=raw.dtype).expand(len(raw))
        normalized = ((raw - self.mean) / self.scale).clamp(-20, 20)
        return torch.cat((normalized, phase.sin()[:, None], phase.cos()[:, None]), -1)

    def forward(self, raw, phase):
        change = self.motion_scale * torch.tanh(self.net(self.features(raw, phase)))
        return torch.maximum(self.low, torch.minimum(self.high, self.base + change))

class Value(nn.Module):
    def __init__(self, selector, widths, task, scale=10.0):
        super().__init__()
        self.register_buffer("mean", selector.mean.detach().clone())
        self.register_buffer("scale", selector.scale.detach().clone())
        self.net = mlp(len(self.mean) + 2, widths, 1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.task = dict(task)
        self.output_scale = scale

    def forward(self, raw, phase):
        phase = torch.as_tensor(phase, device=raw.device, dtype=raw.dtype).expand(len(raw))
        x = torch.cat((((raw - self.mean) / self.scale).clamp(-20, 20),
                       phase.sin()[:, None], phase.cos()[:, None]), -1)
        import math
        beta = -math.log(self.task["gamma_reference"]) / self.task["reference_time"]
        remaining = raw[:, 28].clamp_min(0) * self.task["horizon"]
        stall = self.task["path_speed_weight"] + self.task["path_time_weight"]
        prior = -stall * (-torch.expm1(-beta * remaining)) / beta
        return prior + self.output_scale * torch.tanh(self.net(x).squeeze(-1))

def lambda_targets(rewards, discounts, survival_steps, values, lam):
    """Fixed TD(lambda) labels with interval-specific physical-time discounts."""
    with torch.no_grad():
        future = values[-1].clone()
        result = []
        for j in range(len(rewards) - 1, -1, -1):
            future = rewards[j] + discounts[j] * survival_steps[j] * (
                (1 - lam) * values[j + 1] + lam * future)
            result.append(future)
        return torch.stack(result[::-1])

