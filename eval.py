"""Evaluate one trained job. Evaluation never updates a network or replay buffer."""
import argparse
import json
import os
from pathlib import Path
from bench.protocol import read_job, CONTEXTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--root", default="runs")
    p.add_argument("--checkpoint", default="final", help="final, native_final (released), or a saved decision count")
    p.add_argument("--protocol", choices=["paper", "common"], default="common")
    p.add_argument("--clocks", default="irregular,regular")
    p.add_argument("--contexts", default="nominal", help="Comma-separated contexts; shifts require common")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    job = read_job(a.plan, a.index)
    if a.checkpoint == "native_final" and job["track"] != "released":
        p.error("native_final selects a released periodic checkpoint")
    clocks, contexts = a.clocks.split(","), a.contexts.split(",")
    if a.episodes < 1 or any(c not in ("regular", "irregular") for c in clocks):
        p.error("Use positive episodes and regular/irregular clocks")
    if any(c not in CONTEXTS for c in contexts):
        p.error(f"Contexts: {','.join(CONTEXTS)}")
    if a.protocol == "paper" and contexts != ["nominal"]:
        p.error("Use --protocol common for shifted forces or robot parameters")
    directory = Path(a.root).resolve() / job["run_id"]
    if a.dry_run:
        print(json.dumps(dict(run=job["run_id"], checkpoint=a.checkpoint, protocol=a.protocol,
                             clocks=clocks, contexts=contexts, episodes_per_case=a.episodes,
                             split=a.split, learning_updates=0), indent=2))
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    from bench.evaluation import evaluate_case
    for clock in clocks:
        for context in contexts:
            result = evaluate_case(job, directory, a.checkpoint, a.protocol, clock, context,
                                   a.split, a.episodes, not a.stochastic, a.device)
            print(json.dumps(dict(task=job["task"], algo=job["algo"], seed=job["seed"],
                  clock=clock, context=context,
                  mean_time_score=sum(r["time_score"] for r in result["rows"]) / a.episodes,
                  mean_raw_return=sum(r["raw_decision_return"] for r in result["rows"]) / a.episodes)))


if __name__ == "__main__":
    main()
