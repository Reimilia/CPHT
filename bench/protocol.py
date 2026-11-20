"""Protocol and provenance. This module uses the Python standard library."""
from __future__ import annotations

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


def versions():
    packages = {}
    for name in ("torch", "mujoco", "dm-control", "gymnasium", "stable-baselines3",
                 "sb3-contrib", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)
