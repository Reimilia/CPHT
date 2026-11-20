"""Protocol and provenance. This module uses the Python standard library."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "152162e3633cd634530efb9cc007c7d014cb145a4bbcbdc93b4d141f6d299cbc"
_SOURCE_ROOT = None
TASKS = {
    "cheetah-run": dict(dt=.01, physics_dt=.002, min_dt=.002, max_dt=.03,
                        max_steps=2000, episode_duration=10., tail_p=.99, tail_split=.9,
                        decisions=1_000_000),
    "walker-run": dict(dt=.025, physics_dt=.0025, min_dt=.005, max_dt=.075,
                       max_steps=2000, episode_duration=25., tail_p=.99, tail_split=.9,
                       decisions=1_000_000),
    "humanoid-walk": dict(dt=.025, physics_dt=.005, min_dt=.01, max_dt=.04,
                          max_steps=1000, episode_duration=25., tail_p=.8, tail_split=.5,
                          decisions=2_000_000),
    "quadruped-run": dict(dt=.02, physics_dt=.005, min_dt=.005, max_dt=.05,
                          max_steps=2000, episode_duration=20., tail_p=.99, tail_split=.9,
                          decisions=1_000_000),
}
RELEASED = ("ct_sac", "ct_td3", "q_learning", "cppo", "sac", "td3", "ppo", "trpo")
MATCHED = ("sac_smdp", "ct_sac_corrected", "ct_picard", "tpht", "collocation_direct",
           "picard", "tpht_damping")
CONTEXTS = ("nominal", "damping_1p5", "mass_1p2", "point_drag", "point_pulse")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def source_root():
    global _SOURCE_ROOT
    if _SOURCE_ROOT is not None:
        return _SOURCE_ROOT
    archive = ROOT / "vendor/ct-rl-source.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("The CT-RL source archive does not match the protocol hash.")
    destination = ROOT / "vendor/extracted"
    root = destination / "ct-rl-master"
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix="extract_", dir=archive.parent))
        try:
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    relative = Path(item.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe source archive member")
                z.extractall(staging)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    # Check the extracted files as well. A cached edit must never pass as released code.
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe source archive member")
            if item.is_dir():
                continue
            target = destination / relative
            expected = z.read(item)
            if target.exists():
                if target.read_bytes() != expected:
                    raise RuntimeError(f"Edited upstream source: {relative}")
            else:
                raise RuntimeError(f"Missing upstream source: {relative}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _SOURCE_ROOT = root
    return root


def env_kwargs(task, clock="irregular"):
    t = TASKS[task]
    return dict(time_sampling="irregular" if clock == "irregular" else "uniform",
                dt=t["dt"], physics_dt=t["physics_dt"], min_dt=t["min_dt"],
                max_dt=t["max_dt"], max_steps=t["max_steps"],
                episode_duration=t["episode_duration"], return_reward_increment=False,
                time_sampling_kwargs=dict(dist="two_tail_uniform", tail_p=t["tail_p"],
                                          tail_split=t["tail_split"]))


def released_row(algo, task, rank):
    with (source_root() / "benchmarks/hyperparams" / f"{algo}.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["env_id"] == task and r["mode"] == rank]
    if len(rows) != 1:
        raise ValueError(f"Expected one configuration for {algo}/{task}/{rank}")
    return rows[0]


def make_job(track, task, algo, seed, rank="top", clock="irregular", settings=None):
    if task not in TASKS:
        raise ValueError("Use a control task from TASKS")
    if not 0 <= int(seed) < 3000 or clock not in ("regular", "irregular"):
        raise ValueError("Use a seed in [0,3000) and a regular/irregular clock")
    choices = RELEASED if track == "released" else MATCHED
    if track not in ("released", "matched") or algo not in choices:
        raise ValueError("Unknown track or method")
    job = dict(schema=1, track=track, task=task, algo=algo, seed=int(seed), rank=rank,
               clock=clock, source_sha256=SOURCE_SHA256, task_spec=TASKS[task].copy(),
               decision_budget=TASKS[task]["decisions"])
    if track == "released":
        if clock != "irregular":
            raise ValueError("Released training uses its CSV clock. Use regular clocks at evaluation.")
        job["csv_row"] = released_row(algo, task, rank)
        job["decision_budget"] = int(job["csv_row"]["total_timesteps"])
    else:
        job["settings"] = dict(batch_size=256, replay_capacity=300_000, warmup=10_000,
                               hidden=[400, 300], learning_rate=3e-4,
                               gamma_native=.99 if task == "humanoid-walk" else .98,
                               alpha_rate=.2, polyak=.005, knn=8, kernel_temperature=1.,
                               picard_eta=.5, solver_sigma=.5, solver_damping=1.,
                               solver_omega=3., solver_steps=1, save_every=100_000,
                               log_every=1000, gradient_updates_per_decision=1)
        if settings:
            job["settings"].update(settings)
        if job["settings"]["gradient_updates_per_decision"] != 1:
            raise ValueError("This protocol specifies one actor and one critic update per decision.")
        if not 0 < job["settings"]["gamma_native"] < 1:
            raise ValueError("gamma_native must lie in (0,1)")
        c = job["settings"]
        if min(c["batch_size"], c["replay_capacity"], c["knn"], c["solver_steps"],
               c["save_every"], c["log_every"], c["kernel_temperature"], c["learning_rate"]) <= 0:
            raise ValueError("Counts, kernel temperature, and learning rate must be positive")
        if c["warmup"] < c["batch_size"] or c["replay_capacity"] < c["batch_size"]:
            raise ValueError("Warmup and replay capacity must cover a batch")
        if not 0 < c["picard_eta"] <= 1 or not 0 < c["polyak"] <= 1 or c["alpha_rate"] <= 0:
            raise ValueError("Invalid relaxation, target averaging, or entropy temperature")
        if c["solver_sigma"] <= 0 or c["solver_damping"] <= 0:
            raise ValueError("The phase solver requires positive solver time and damping")
    job["run_id"] = f"{track}/{task}/{algo}/seed_{seed}_{digest(job)[:10]}"
    return job


def read_job(plan, index):
    lines = [s for s in Path(plan).read_text().splitlines() if s.strip()]
    job = json.loads(lines[index])
    if job["source_sha256"] != SOURCE_SHA256 or job["task"] not in TASKS:
        raise ValueError("Plan provenance or task mismatch")
    rebuilt = make_job(job["track"], job["task"], job["algo"], job["seed"],
                       job["rank"], job["clock"], job.get("settings"))
    if rebuilt != job:
        raise ValueError("Plan was modified after construction. Generate it again.")
    return job


def versions():
    packages = {}
    for name in ("torch", "mujoco", "dm-control", "gymnasium", "stable-baselines3",
                 "sb3-contrib", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)
