"""Matched training, atomic checkpoints, and episode-boundary recovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from .env import entropy_weight, make_common_env
from .learner import Learner, Replay
from .protocol import write_json, versions


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_matched(job, directory, device, resume=False, walltime_hours=None):
    directory = Path(directory)
    cfg = job["settings"]
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    torch.backends.cudnn.benchmark = False
    counters = dict(decisions=0, physics_steps=0, simulated_seconds=0., updates=0,
                    actor_steps=0, critic_steps=0, linear_solves=0, operator_matvecs=0,
                    knn_queries=0, distance_pairs=0, actor_forward_batches=0, critic_forward_batches=0,
                    update_seconds=0., environment_seconds=0., training_wall_seconds=0.)
    episode = 0
    warmup_rng = np.random.default_rng(job["seed"] + 11)
    collection_generator = torch.Generator(device=device).manual_seed(job["seed"] + 29)

    def new_env(number):
        # Independent episode seeds allow exact recovery at an episode boundary.
        start = time.perf_counter()
        episode_seed = 1_000_000_000 + 1_000_000 * job["seed"] + number
        env = make_common_env(job["task"], episode_seed, job["clock"],
                              gamma_native=cfg["gamma_native"])
        observation, _ = env.reset(seed=episode_seed)
        counters["environment_seconds"] += time.perf_counter() - start
        return env, observation

    env, obs = new_env(episode)
    native_dt = env.dt_default
    entropy_scale = entropy_weight(job["task"], native_dt, cfg["gamma_native"], job["clock"])
    learner = Learner(len(obs), int(np.prod(env.action_space.shape)), job, device, entropy_scale)
    replay = Replay(cfg["replay_capacity"], len(obs), learner.dimensions["action_dim"], job["seed"] + 17)
    restored = None
    if resume:
        # latest.pt contains NumPy replay arrays. Load only checkpoints you created.
        restored = torch.load(directory / "latest.pt", map_location=device, weights_only=False)
        if restored["job"] != job or not restored["episode_boundary"]:
            raise ValueError("Resume requires the same plan and an episode-boundary checkpoint")
        learner.load(restored["learner"])
        replay.load(restored["replay"])
        counters, episode = restored["counters"], restored["next_episode"]
        warmup_rng.bit_generator.state = restored["warmup_rng"]
        collection_generator.set_state(restored["collection_rng"].cpu())
        env.close()
        env, obs = new_env(episode)
        # Restore learning RNGs after all constructors have run.
        torch.set_rng_state(restored["torch_rng"].cpu())
        if torch.cuda.is_available() and restored["cuda_rng"]:
            torch.cuda.set_rng_state_all([v.cpu() for v in restored["cuda_rng"]])
        np.random.set_state(restored["numpy_rng"])
        random.setstate(restored["python_rng"])
        log = directory / "train.jsonl"
        if log.exists():
            kept = [s for s in log.read_text().splitlines() if json.loads(s)["decisions"] <= counters["decisions"]]
            log.write_text("\n".join(kept) + ("\n" if kept else ""))
        for path in directory.glob("policy_*.pt"):
            if int(path.stem.split("_")[-1]) > counters["decisions"]:
                path.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run.json", dict(job=job, versions=versions(), device=str(device),
               gpu=torch.cuda.get_device_name(learner.device) if learner.device.type == "cuda" else None,
               cuda=torch.version.cuda, torch_threads=torch.get_num_threads(),
               native_dt=native_dt, entropy_weight=entropy_scale, scope="sampled finite Q operator + neural distillation"))
    stop = dict(requested=False)

    def request_stop(*_):
        stop["requested"] = True
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        signal.signal(sig, request_stop)
    started = time.perf_counter()
    previous_wall = counters["training_wall_seconds"]
    last_save = counters["decisions"]

    def stamp():
        counters["training_wall_seconds"] = previous_wall + time.perf_counter() - started

    def save_policy(path):
        stamp()
        atomic_torch_save(dict(job=job, learner=learner.state(False), counters=counters.copy()), path)

    def save_boundary(next_episode):
        stamp()
        atomic_torch_save(dict(job=job, learner=learner.state(), replay=replay.state(),
             counters=counters.copy(), next_episode=next_episode, episode_boundary=True,
             warmup_rng=warmup_rng.bit_generator.state, collection_rng=collection_generator.get_state(),
             torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             numpy_rng=np.random.get_state(), python_rng=random.getstate()), directory / "latest.pt")

    with (directory / "train.jsonl").open("a", buffering=1) as log:
        while counters["decisions"] < job["decision_budget"]:
            if counters["decisions"] < cfg["warmup"]:
                normalized_action = warmup_rng.uniform(-1., 1., env.action_space.shape).astype(np.float32)
            else:
                normalized_action = learner.act(obs, deterministic=False, generator=collection_generator)
            actual_action = env.action_space.low + (normalized_action + 1) * (env.action_space.high - env.action_space.low) / 2
            start = time.perf_counter()
            next_obs, reward, term, trunc, info = env.step(actual_action)
            counters["environment_seconds"] += time.perf_counter() - start
            dt = float(info.get("dt_used", 0.))
            if dt <= 0:
                raise RuntimeError("A common-protocol transition must advance physical time")
            replay.add(obs=obs, actions=normalized_action, next_obs=next_obs, reward=reward,
                       endpoint_rate=info["endpoint_reward_rate"], dt=dt, terminated=float(term))
            counters["decisions"] += 1
            counters["physics_steps"] += info["physics_steps"]
            counters["simulated_seconds"] += dt
            stats = {}
            log_now = counters["decisions"] % cfg["log_every"] == 0
            if counters["decisions"] >= cfg["warmup"] and replay.size >= cfg["batch_size"]:
                if learner.device.type == "cuda":
                    torch.cuda.synchronize(learner.device)
                start = time.perf_counter()
                batch = replay.sample(cfg["batch_size"], learner.device, native_dt)
                stats = learner.update(batch, diagnostics=log_now)
                if learner.device.type == "cuda":
                    torch.cuda.synchronize(learner.device)
                counters["update_seconds"] += time.perf_counter() - start
                for key in ("linear_solves", "operator_matvecs", "knn_queries", "distance_pairs",
                            "actor_forward_batches", "critic_forward_batches"):
                    counters[key] += stats.get(key, 0)
                for key in ("updates", "actor_steps", "critic_steps"):
                    counters[key] += 1
            obs = next_obs
            if log_now:
                stamp()
                record = dict(counters, **{k: v for k, v in stats.items() if k not in counters})
                if any(not np.isfinite(v) for v in record.values() if isinstance(v, (int, float))):
                    raise FloatingPointError("Non-finite learning diagnostic")
                log.write(json.dumps(record) + "\n")
            if counters["decisions"] % cfg["save_every"] == 0:
                save_policy(directory / f"policy_{counters['decisions']:09d}.pt")
            if walltime_hours and time.perf_counter() - started >= walltime_hours * 3600:
                stop["requested"] = True
            if term or trunc:
                episode += 1
                if stop["requested"] or counters["decisions"] - last_save >= cfg["save_every"]:
                    save_boundary(episode)
                    last_save = counters["decisions"]
                env.close()
                if stop["requested"]:
                    break
                if counters["decisions"] < job["decision_budget"]:
                    env, obs = new_env(episode)
    stamp()
    complete = counters["decisions"] >= job["decision_budget"]
    if complete:
        save_policy(directory / "final.pt")
    env.close()
    result = dict(status="complete" if complete else "paused", counters=counters,
                  primary_checkpoint="final.pt" if complete else None,
                  resume_checkpoint="latest.pt" if (directory / "latest.pt").exists() else None,
                  elapsed_episode_count=episode)
    write_json(directory / "status.json", result)
    return result
