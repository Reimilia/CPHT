"""Train exactly one indexed run. Use --dry-run to inspect it without dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback
from bench.protocol import ROOT, read_job, source_root, versions, write_json


def train_released(job, directory):
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    root = source_root()
    module = "benchmarks.run_ct_rl" if job["algo"] in ("ct_sac", "ct_td3", "q_learning", "cppo") else "benchmarks.run_discrete_rl"
    final_ct = directory / "native_final.pth"
    command = [sys.executable, "-u", str(ROOT / "released_entry.py"),
               "--family", "ct" if module.endswith("run_ct_rl") else "discrete",
               "--final", str(final_ct), "--expected-budget", str(job["decision_budget"]),
               "--", "--algos", job["algo"],
               "--env_id", job["task"], "--mode", job["rank"], "--seed", str(job["seed"]),
               "--hyperparams_dir", str(root / "benchmarks/hyperparams"),
               "--log_root", str(directory / "native_logs"),
               "--save_root", str(directory / "native_models"),
               "--total_timesteps", str(job["decision_budget"]), "--n_eval_episodes", "10",
               "--desc", "frozen_source"]
    write_json(directory / "run.json", dict(job=job, versions=versions(), command=command,
               scope="released algorithms and CSV settings; added post-training CT save callback", device="auto"))
    started = time.perf_counter()
    with (directory / "native_stdout.log").open("w") as log:
        completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Native runner failed. Read {directory / 'native_stdout.log'}")
    if module.endswith("run_discrete_rl"):
        paths = list((directory / "native_models").rglob("final_model.zip"))
    else:
        paths = list((directory / "native_models").rglob(f"*_{job['decision_budget']}_steps.pth"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one final-budget checkpoint, found {len(paths)}. Inspect native_stdout.log.")
    primary = paths[0] if module.endswith("run_discrete_rl") else final_ct
    if not primary.exists():
        raise RuntimeError("Post-training final checkpoint is missing")
    result = dict(status="complete", primary_checkpoint=str(primary.relative_to(directory)),
                  native_budget_checkpoint=str(paths[0].relative_to(directory)),
                  training_wall_seconds=time.perf_counter() - started,
                  decision_budget=job["decision_budget"], note="Native time includes inline validation and I/O; final CT save is after the last update.")
    write_json(directory / "status.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--root", default="runs")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--resume", action="store_true", help="Matched track only; restores latest.pt")
    p.add_argument("--walltime-hours", type=float, help="Matched track: request pause at the next episode boundary")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    job = read_job(a.plan, a.index)
    directory = Path(a.root).resolve() / job["run_id"]
    if a.dry_run:
        print(json.dumps(dict(job=job, directory=str(directory), device=a.device), indent=2))
        return
    if job["track"] == "released" and (a.resume or a.walltime_hours):
        p.error("The released checkpoint format stores model weights. Resume and boundary pause are supported by the matched track.")
    if (directory / "status.json").exists():
        status = json.loads((directory / "status.json").read_text())
        if status["status"] == "complete":
            print(f"Already complete: {directory}")
            return
    if directory.exists() and any(directory.iterdir()) and not a.resume:
        p.error("Run directory already contains an attempt. Use --resume for a matched checkpoint, or select a new --root.")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        p.error("This run has a lock. Check the active job before removing a stale run.lock.")
    with os.fdopen(descriptor, "w") as f:
        f.write(str(os.getpid()) + "\n")
    # This affects only this process and its child. CUDA selection is logged.
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        if job["track"] == "released":
            result = train_released(job, directory)
        else:
            from bench.training import train_matched
            result = train_matched(job, directory, a.device, a.resume, a.walltime_hours)
    except Exception as exc:
        write_json(directory / "status.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink(missing_ok=True)
    print(json.dumps(dict(directory=str(directory), **result), indent=2))


if __name__ == "__main__":
    main()
