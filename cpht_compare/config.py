"""Validated sweep configuration. Planning requires only the standard library."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path

METHODS = ("cem", "shooting", "selector", "selector_value")
CONDITIONS = ("nominal", "layout", "friction_change", "sensor_shift", "steps")
DEFAULTS = {
    "schema": "cpht_comparison_v1",
    "cases": ["F", "J"], "seeds": [0, 1, 2],
    "information_modes": ["state_belief"],
    "methods": list(METHODS), "horizons_seconds": [0.2, 0.4, 0.6],
    "terrain": "terrain", "smoke_only": False,
    "learning": {
        "total_decisions": 100000, "calibration_decisions": 4096,
        "calibration_episode_steps": 32, "capacity": 300000,
        "hidden": [256, 256], "batch": 256,
        "save_every": 10000, "log_every": 1000,
        "pure": {"contact_supervision": "aggregate_ports"},
        "sensing": {"sensor_period": 0.02, "sensor_delay": 0.02,
                    "sensor_dropout": 0.05, "friction_events": []},
    },
    "task": {
        "control_task": "path_speed", "contact_backend": "ct_terrain",
        "contact_fixed_seed": 1, "clock": "fixed",
        "contact_profile": "mixed", "terrain_elevation": "hills",
    },
    "optimization": {
        "shooting_steps": 8, "shooting_lr": 0.02,
        "selector_hidden": [128, 128], "selector_lr": 0.0003,
        "batch_size": 16, "update_every": 8, "updates_per_block": 1,
        "grad_clip": 1.0, "critic_epochs": 2, "gae_lambda": 0.95,
        "critic_scale": 100.0, "failure_softness": 0.04,
        "action_noise": 0.05, "max_replay": 24000,
        "prediction_error_limit": 5.0, "solver_residual_limit": 0.1,
        "max_grad_norm": 10000.0, "contact_mode": "piecewise",
    },
    "evaluation": {
        "episodes": 5, "clocks": ["fixed", "irregular"],
        "conditions": ["nominal", "layout", "friction_change", "sensor_shift", "steps"],
        "test_cases": ["same", "J"], "max_decisions": None, "videos": 0,
    },
}

def merge(base, changes, prefix=""):
    result = deepcopy(base)
    if prefix in ("task", "learning.pure", "learning.sensing"):
        result.update(deepcopy(changes))
        return result
    unknown = set(changes) - set(base)
    if unknown:
        raise ValueError(f"Unknown comparison fields: {sorted(unknown)}")
    for key, value in changes.items():
        sub = f"{prefix}.{key}" if prefix else key
        result[key] = merge(base[key], value, sub) if isinstance(base[key], dict) else deepcopy(value)
    return result

def load(path=None):
    p = merge(DEFAULTS, json.loads(Path(path).read_text()) if path else {})
    if p["schema"] != DEFAULTS["schema"]:
        raise ValueError("Unsupported comparison schema")
    for name, allowed in (
        ("cases", ("F", "J")), ("methods", METHODS),
        ("information_modes", ("state", "state_belief", "sensed_belief")),
    ):
        if not p[name] or len(set(p[name])) != len(p[name]) or any(x not in allowed for x in p[name]):
            raise ValueError(f"Invalid {name}: {p[name]}")
    if not p["seeds"] or any(type(x) is not int or not 0 <= x < 3000 for x in p["seeds"]):
        raise ValueError("Seeds must be integers in [0,3000)")
    if len(set(p["seeds"])) != len(p["seeds"]):
        raise ValueError("Repeated training seeds")
    if not p["horizons_seconds"] or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
        for x in p["horizons_seconds"]
    ):
        raise ValueError("Positive physical rollout horizons required")
    if len(set(p["horizons_seconds"])) != len(p["horizons_seconds"]):
        raise ValueError("Repeated physical horizons")
    if p["terrain"] not in ("hills", "friction", "terrain", "slopes", "steps"):
        raise ValueError("Unknown terrain")
    o = p["optimization"]
    counts = ("shooting_steps", "batch_size", "update_every", "updates_per_block",
              "critic_epochs", "max_replay")
    for name in counts:
        if type(o[name]) is not int or o[name] < 1:
            raise ValueError(f"Positive integer required: {name}")
    if o["contact_mode"] != "piecewise":
        raise ValueError("Only the audited piecewise contact derivative is implemented")
    if not o["selector_hidden"] or any(type(x) is not int or x < 1 for x in o["selector_hidden"]):
        raise ValueError("Positive selector hidden-layer widths required")
    for name in ("shooting_lr", "selector_lr", "grad_clip", "critic_scale",
                 "failure_softness", "prediction_error_limit", "solver_residual_limit", "max_grad_norm"):
        if not math.isfinite(o[name]) or o[name] <= 0:
            raise ValueError(f"Positive finite parameter required: {name}")
    if not 0 <= o["action_noise"] < 1 or not 0 <= o["gae_lambda"] <= 1:
        raise ValueError("Invalid action noise or GAE lambda")
    e = p["evaluation"]
    if not e["clocks"] or any(x not in ("fixed", "irregular") for x in e["clocks"]):
        raise ValueError("Invalid evaluation clocks")
    if not e["conditions"] or any(x not in CONDITIONS for x in e["conditions"]):
        raise ValueError("Invalid evaluation conditions")
    if not e["test_cases"] or any(x not in ("same", "F", "J") for x in e["test_cases"]):
        raise ValueError("Invalid test cases")
    for name in ("episodes",):
        if type(e[name]) is not int or e[name] < 1:
            raise ValueError(f"Positive integer required: evaluation.{name}")
    if e["max_decisions"] is not None and (type(e["max_decisions"]) is not int or e["max_decisions"] < 1):
        raise ValueError("Positive optional evaluation cap required")
    if type(e["videos"]) is not int or e["videos"] < 0:
        raise ValueError("Nonnegative video count required")
    return p

def jobs(profile):
    for mode, case, seed, horizon, method in itertools.product(
        profile["information_modes"], profile["cases"], profile["seeds"],
        profile["horizons_seconds"], profile["methods"]
    ):
        yield dict(mode=mode, case=case, seed=seed, horizon=float(horizon), method=method)

def settings_for(profile, job):
    from physical_fj.config import settings
    values = deepcopy(profile["learning"])
    values["sensing"]["mode"] = job["mode"]
    return settings(job["case"], profile["terrain"], job["seed"], **values)

def control_for(profile, s):
    from fjbench.config import make_config
    task = deepcopy(profile["task"])
    task.update(calibration_decisions=s["calibration_decisions"],
                total_decisions=s["total_decisions"], batch=min(s["batch"], s["calibration_decisions"]))
    return make_config(s["case"], "contact", s["seed"], task)

def paths(profile, job, run_root="runs/cpht_comparison", artifact_root="artifacts/cpht_comparison"):
    from bench.protocol import digest
    from physical_fj.config import calibration_id
    s = settings_for(profile, job)
    c = control_for(profile, s)
    # Reuse the existing validator for the physical task and sensor contract.
    # make_config validates task fields; extra sensor checks need no temporary file.
    if len(c["contact_offsets"]) != 7 or s["sensing"]["sensor_period"] < c["physics_dt"] - 1e-12:
        raise ValueError("Seven scan locations and sensor period >= physics timestep required")
    if c["contact_robot_variation"]:
        raise ValueError("This comparison currently keeps robot topology/parameters fixed")
    ratio = job["horizon"] / s["pure"]["planning_dt"]
    if ratio < 1 - 1e-8 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("Choose a horizon that is a positive integer multiple of planning_dt")
    identity = calibration_id(s, c)
    artifact = Path(artifact_root) / job["mode"] / job["case"] / f"seed{job['seed']}_{identity}"
    signature = digest(dict(schema=profile["schema"], s=s, c=c, method=job["method"],
                            horizon=job["horizon"], optimization=profile["optimization"]))[:12]
    horizon = f"h{round(1000 * job['horizon']):04d}ms"
    run = Path(run_root) / job["mode"] / job["case"] / horizon / job["method"] / f"seed{job['seed']}_{signature}"
    return s, c, artifact, run

def evaluation_jobs(profile, job):
    for case, clock, condition in itertools.product(
        profile["evaluation"]["test_cases"], profile["evaluation"]["clocks"], profile["evaluation"]["conditions"]
    ):
        selected = job["case"] if case == "same" else case
        if condition == "sensor_shift" and job["mode"] != "sensed_belief":
            continue
        yield selected, clock, condition

def plan(profile, run_root, artifact_root):
    output = []
    for job in jobs(profile):
        s, c, artifact, run = paths(profile, job, run_root, artifact_root)
        tests = list(dict.fromkeys(evaluation_jobs(profile, job)))
        output.append(dict(job, artifact=str(artifact), run=str(run),
                           calibration_decisions=s["calibration_decisions"],
                           total_decisions=s["total_decisions"],
                           eval_conditions=len(tests), physical_horizon_seconds=job["horizon"],
                           learning_clock=c["clock"]))
    return output

