"""Shared neural training code. Heavy imports occur only for train/eval work."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .operators import torch_collocation


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


class Learner:
    def __init__(self, obs_dim, action_dim, job, device, entropy_weight):
        self.job, self.cfg, self.device = job, job["settings"], torch.device(device)
        self.entropy_weight = entropy_weight
        self.actor = Actor(obs_dim, action_dim, self.cfg["hidden"]).to(self.device)
        self.critics = nn.ModuleList([mlp(obs_dim + action_dim, 1, self.cfg["hidden"]) for _ in range(2)]).to(self.device)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.cfg["learning_rate"])
        self.dimensions = dict(obs_dim=obs_dim, action_dim=action_dim)
        self.updates = 0

    def q(self, obs, actions, target=False):
        net = self.targets if target else self.critics
        value = torch.cat((obs, actions), dim=1)
        return torch.stack([n(value) for n in net], dim=0)

    def act(self, obs, deterministic=True, generator=None):
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
            return self.actor.sample(tensor, deterministic, generator)[0][0].cpu().numpy()

    def update(self, batch, diagnostics=False):
        obs, actions, next_obs = batch["obs"], batch["actions"], batch["next_obs"]
        c, method = self.cfg, self.job["algo"]
        u = batch["dt"] / batch["native_dt"]
        beta = -np.log(c["gamma_native"])
        discount = torch.exp(-beta * u) * (1 - batch["terminated"])
        alpha = c["alpha_rate"]
        scale = 1. if method == "ct_sac_corrected" else self.entropy_weight
        stats = dict(linear_solves=0, operator_matvecs=0, knn_queries=0, distance_pairs=0)
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_obs)
            next_q = self.q(next_obs, next_actions, target=True).min(0).values
            if method == "sac_smdp":
                target = batch["reward"] + discount * (next_q - alpha * scale * next_logp)
            elif method == "ct_sac_corrected":
                current_actions, current_logp = self.actor.sample(obs)
                current_v = self.q(obs, current_actions, target=True).min(0).values - alpha * current_logp
                next_v = next_q - alpha * next_logp
                target = batch["endpoint_rate"] + current_v + (discount * next_v - current_v) / u
            else:
                z = self.q(obs, actions, target=True).min(0).values
                target, stats = torch_collocation(
                    obs, actions, next_obs, next_actions, batch["reward"], discount,
                    alpha * scale * next_logp, z, next_q, c, method,
                    collect_stats=diagnostics)
        prediction = self.q(obs, actions)
        # Match the released ensemble convention: sum the two mean losses.
        critic_loss = ((prediction - target.unsqueeze(0)) ** 2).mean(dim=(1, 2)).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critics.requires_grad_(False)
        sampled_actions, logp = self.actor.sample(obs)
        actor_loss = (alpha * scale * logp - self.q(obs, sampled_actions).min(0).values).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)
        with torch.no_grad():
            if diagnostics:
                fitted = self.q(obs, actions)
                stats.update(critic_loss=float(critic_loss), actor_loss=float(actor_loss),
                             max_critic_fit_error=float((fitted - target.unsqueeze(0)).abs().max()),
                             target_abs_max=float(target.abs().max()))
            for target_parameter, parameter in zip(self.targets.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, c["polyak"])
        self.updates += 1
        stats["actor_forward_batches"] = 3 if method == "ct_sac_corrected" else 2
        stats["critic_forward_batches"] = (3 if method == "sac_smdp" else 4) + int(diagnostics)
        return stats

    def state(self, optimizers=True):
        result = dict(actor=self.actor.state_dict(), critics=self.critics.state_dict(),
                      targets=self.targets.state_dict(), dimensions=self.dimensions,
                      entropy_weight=self.entropy_weight, updates=self.updates)
        if optimizers:
            result.update(actor_optimizer=self.actor_optimizer.state_dict(),
                          critic_optimizer=self.critic_optimizer.state_dict())
        return result

    def load(self, state, optimizers=True):
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.targets.load_state_dict(state["targets"])
        self.updates = state["updates"]
        if optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])


class Replay:
    def __init__(self, capacity, obs_dim, action_dim, seed):
        self.arrays = {k: np.empty((capacity, d), np.float32) for k, d in
                       dict(obs=obs_dim, next_obs=obs_dim, actions=action_dim,
                            reward=1, endpoint_rate=1, dt=1, terminated=1).items()}
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, **transition):
        for k, array in self.arrays.items():
            array[self.position] = transition[k]
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, device, native_dt):
        indices = self.rng.integers(self.size, size=count)
        batch = {k: torch.as_tensor(v[indices], device=device) for k, v in self.arrays.items()}
        batch["native_dt"] = native_dt
        return batch

    def state(self):
        return dict(arrays={k: v[:self.size].copy() for k, v in self.arrays.items()},
                    capacity=self.capacity, position=self.position, size=self.size,
                    rng=self.rng.bit_generator.state)

    def load(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from the checkpoint")
        self.position, self.size = state["position"], state["size"]
        for k, v in state["arrays"].items():
            self.arrays[k][:self.size] = v
        self.rng.bit_generator.state = state["rng"]
