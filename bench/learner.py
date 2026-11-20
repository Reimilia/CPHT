"""MLP and tanh-Gaussian actor shared by the CT-SAC baseline."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim, output_dim, hidden):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(input_dim, size), nn.ReLU()))
        input_dim = size
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 2 * action_dim, hidden)

    def sample(self, obs, deterministic=False, generator=None):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5., 2.).exp())
        raw = mean if deterministic else mean + distribution.scale * torch.randn(
            mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        action = raw.tanh()
        log_prob = (distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(1, keepdim=True)
        return action, log_prob


