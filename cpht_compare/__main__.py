"""One entry point for the differentiable CPHT comparison."""
import argparse
import json
import os
from pathlib import Path
import sys
from . import config

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "prepare", "train", "eval", "report", "all", "gradcheck"))
    p.add_argument("--profile", default="cases/cpht_comparison.json")
    p.add_argument("--run-root", default="runs/cpht_comparison")
    p.add_argument("--artifacts", default="artifacts/cpht_comparison")
    p.add_argument("--methods", help="Comma-separated subset")
    p.add_argument("--cases", help="F,J")
    p.add_argument("--seeds", help="Comma-separated training seeds")
    p.add_argument("--information", help="Comma-separated information modes")
    p.add_argument("--horizons", help="Comma-separated physical seconds")
    p.add_argument("--conditions", help="Comma-separated evaluation conditions")
    p.add_argument("--clocks", help="fixed,irregular")
    p.add_argument("--test-cases", help="same,F,J")
    p.add_argument("--episodes", type=int)
    p.add_argument("--eval-max-decisions", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gl", default="egl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--output", default="reports/cpht_comparison.json")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    profile = config.load(args.profile)
    for arg, key, cast in (("methods", "methods", str), ("cases", "cases", str),
                           ("seeds", "seeds", int), ("information", "information_modes", str),
                           ("horizons", "horizons_seconds", float)):
        if getattr(args, arg):
            profile[key] = [cast(x.strip()) for x in getattr(args, arg).split(",") if x.strip()]
    for arg, key in (("conditions", "conditions"), ("clocks", "clocks"), ("test_cases", "test_cases")):
        if getattr(args, arg):
            profile["evaluation"][key] = getattr(args, arg).split(",")
    if args.episodes is not None:
        profile["evaluation"]["episodes"] = args.episodes
    if args.eval_max_decisions is not None:
        profile["evaluation"]["max_decisions"] = args.eval_max_decisions
    # Revalidate overrides without creating an intermediate profile file.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(profile, f)
        f.flush()
        profile = config.load(f.name)
    planned = config.plan(profile, args.run_root, args.artifacts)
    if args.stage == "plan" or args.dry_run:
        print(json.dumps(dict(stage=args.stage, dry_run=True, jobs=planned,
                              training_runs=len(planned),
                              shared_calibrations=len({x["artifact"] for x in planned}),
                              expensive_training_executed=False), indent=2))
        return 0
    os.environ.setdefault("MUJOCO_GL", args.gl)
    if args.stage == "report":
        from .report import collect
        result = collect(profile, args.run_root, args.artifacts, args.output)
        print(json.dumps(dict(report=args.output, reports=result["report_count"], missing=result["missing_count"])))
        return 0
    try:
        from . import pipeline
    except ImportError as exc:
        raise SystemExit(f"Missing runtime dependency: {exc}. Activate Torch and install requirements-physical-first.txt") from exc
    failures = []
    prepared = set()
    gradients = set()
    stages = ("prepare", "train", "eval", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "report":
            from .report import collect
            result = collect(profile, args.run_root, args.artifacts, args.output)
            print(json.dumps(dict(stage=stage, output=args.output, reports=result["report_count"], missing=result["missing_count"])))
            continue
        for job in config.jobs(profile):
            _, _, artifact, run = config.paths(profile, job, args.run_root, args.artifacts)
            try:
                if stage == "prepare":
                    if str(artifact) in prepared:
                        continue
                    result = pipeline.preparation(profile, job, args.run_root, args.artifacts)
                    prepared.add(str(artifact))
                    print(json.dumps(dict(stage=stage, artifact=str(artifact), model_sha256=result["model_sha256"])))
                elif stage == "train":
                    result = pipeline.training(profile, job, args.run_root, args.artifacts,
                                               args.device, args.resume)
                    print(json.dumps(dict(stage=stage, job=job, result=result)))
                elif stage == "gradcheck":
                    key = (str(artifact), job["horizon"])
                    if key in gradients:
                        continue
                    result = pipeline.gradient_check(profile, job, args.run_root, args.artifacts, args.device)
                    gradients.add(key)
                    print(json.dumps(dict(stage=stage, job=job, result=result)))
                    if not result["finite_gradient"]:
                        raise FloatingPointError("Gradient check produced nonfinite derivative")
                elif stage == "eval":
                    for case, clock, condition in dict.fromkeys(config.evaluation_jobs(profile, job)):
                        result = pipeline.evaluate(run / "final.pt", case, clock, condition,
                                                   args.device, args.overwrite, profile["evaluation"])
                        print(json.dumps(dict(stage=stage, job=job, test_case=case, clock=clock,
                                              condition=condition, errors=sum(bool(x["error"]) for x in result["rows"]))))
            except Exception as exc:
                failure = dict(stage=stage, job=job, type=type(exc).__name__, message=str(exc))
                failures.append(failure)
                print(json.dumps(failure), file=sys.stderr)
                destination = Path(args.run_root) / "stage_failures.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(failures, indent=2) + "\n")
                if not args.keep_going:
                    raise
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())

