"""Frozen policies, paired reset seeds, and separate score definitions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .env import make_common_env, make_paper_env
from .protocol import source_root, write_json, TASKS


def checkpoint_path(job, directory, tag):
    directory = Path(directory)
    if job["track"] == "matched":
        return directory / ("final.pt" if tag == "final" else f"policy_{int(tag):09d}.pt")
    if tag in ("final", "native_final"):
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError("The released run has no completed final-budget checkpoint")
        return directory / status["primary_checkpoint" if tag == "final" else "native_budget_checkpoint"]
    suffix = "pth" if job["algo"] in ("ct_sac", "ct_td3", "q_learning", "cppo") else "zip"
    found = list((directory / "native_models").rglob(f"*_{int(tag)}_steps.{suffix}"))
    if len(found) != 1:
        raise ValueError(f"Expected one checkpoint at {tag}, found {len(found)}")
    return found[0]


def load_policy(job, checkpoint, env, device):
    if job["track"] == "matched":
        from .learner import Learner
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["job"] != job:
            raise ValueError("Policy checkpoint and plan differ")
        s = state["learner"]
        model = Learner(**s["dimensions"], job=job, device=device, entropy_weight=s["entropy_weight"])
        model.load(s, optimizers=False)
        model.actor.eval()

        def act(obs, deterministic, generator):
            action = model.act(obs, deterministic, generator)
            return env.action_space.low + (action + 1) * (env.action_space.high - env.action_space.low) / 2
        return act, state["counters"]
    source_root()
    algo = job["algo"]
    if algo in ("sac", "td3", "ppo", "trpo"):
        from stable_baselines3 import SAC, TD3, PPO
        from sb3_contrib import TRPO
        model = dict(sac=SAC, td3=TD3, ppo=PPO, trpo=TRPO)[algo].load(checkpoint, device=device)
        return lambda obs, deterministic, generator: model.predict(obs, deterministic=deterministic)[0], {}
    from common.utils import load_ct_hyperparams_from_table
    from models import ActorQCriticModel, ActorVCriticModel, CoupledVqModel
    _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
        algo, job["task"], job["rank"], source_root() / "benchmarks/hyperparams")
    model_class = {"ct_sac": ActorQCriticModel, "ct_td3": ActorQCriticModel,
                   "q_learning": CoupledVqModel, "cppo": ActorVCriticModel}[algo]
    model = model_class(observation_space=env.observation_space, action_space=env.action_space,
                        device=device, **model_kwargs)
    model.load_state(str(checkpoint))

    def act(obs, deterministic, generator):
        del generator
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
            return model.act(x, deterministic=deterministic)[0][0].cpu().numpy()
    return act, {}


def evaluate_case(job, directory, tag, protocol, clock, context, split, episodes, deterministic, device):
    if protocol == "paper" and context != "nominal":
        raise ValueError("The paper protocol uses its original environment. Use common for force shifts.")
    checkpoint = checkpoint_path(job, directory, tag)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    factory = make_paper_env if protocol == "paper" else make_common_env
    start_seed = 50_000 if split == "validation" else 100_000
    rows, policy, counters = [], None, {}
    for episode in range(episodes):
        episode_seed = start_seed + episode
        kwargs = {} if protocol == "paper" else dict(context=context,
                    gamma_native=job.get("settings", {}).get("gamma_native", .98))
        # Rebuild DMC each episode. The released wrapper does not reseed task
        # physics when reset(seed=...) is called on an existing instance.
        env = factory(job["task"], episode_seed, clock, **kwargs)
        try:
            obs, _ = env.reset(seed=episode_seed)
            if policy is None:
                policy, counters = load_policy(job, checkpoint, env, device)
                if job["track"] == "released":
                    status = json.loads((Path(directory) / "status.json").read_text())
                    counters = dict(decisions=job["decision_budget"] if tag in ("final", "native_final") else int(tag),
                                    training_wall_seconds=status.get("training_wall_seconds"),
                                    time_includes_inline_validation=True)
            torch.manual_seed(episode_seed)
            np.random.seed(episode_seed)
            generator = torch.Generator(device=device).manual_seed(episode_seed)
            raw, integral, elapsed, work = 0., 0., 0., 0.
            steps, physics_steps, active_dofs, max_force = 0, 0, 0, 0.
            while True:
                action = np.asarray(policy(obs, deterministic, generator), np.float32)
                if not np.isfinite(action).all():
                    raise FloatingPointError("Policy returned a non-finite action")
                obs, reward, terminated, truncated, info = env.step(np.clip(action, env.action_space.low, env.action_space.high))
                dt = float(info.get("dt_used", 0.))
                if dt <= 0:
                    if truncated:
                        break
                    raise RuntimeError("Evaluation transition did not advance time")
                if info.get("physics_error"):
                    raise RuntimeError("Native MuJoCo physics error")
                rate = float(info.get("endpoint_reward_rate", reward))
                raw += rate
                integral += float(info.get("reward_time_integral", rate * dt))
                elapsed += dt
                steps += 1
                physics_steps += int(info.get("physics_steps", round(dt / env.physics_dt)))
                work += float(info.get("external_work", 0.))
                active_dofs = max(active_dofs, info.get("directly_forced_dofs", 0))
                max_force = max(max_force, info.get("max_generalized_force", 0.))
                if terminated or truncated:
                    break
                if steps > TASKS[job["task"]]["max_steps"] + 2:
                    raise RuntimeError("Evaluation exceeded the control-step cap")
            horizon = TASKS[job["task"]]["episode_duration"]
            row = dict(episode_seed=episode_seed, raw_decision_return=raw,
                       time_score=1000 * integral / horizon, reward_time_integral=integral,
                       elapsed_seconds=elapsed, horizon_coverage=elapsed / horizon,
                       decisions=steps, physics_steps=physics_steps, external_work=work,
                       directly_forced_dofs=active_dofs, max_generalized_force=max_force)
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError("Non-finite evaluation metric")
            rows.append(row)
        finally:
            env.close()
    mode = "deterministic" if deterministic else "stochastic"
    result = dict(job=job, checkpoint=tag, checkpoint_sha256=checkpoint_sha,
                  protocol=protocol, clock=clock, context=context, split=split,
                  policy_mode=mode, episodes=episodes, train_counters=counters, rows=rows)
    filename = f"{protocol}_{tag}_{split}_{mode}_{clock}_{context}_n{episodes}.json"
    write_json(Path(directory) / "evaluations" / filename, result)
    return result
