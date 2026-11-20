"""One entry point for the CPHT experiment: calibrate, train, evaluate, report, verify.

Stages
  plan       list the training runs a profile implies (no computation)
  prepare    acquire instrumented calibration data and fit the pH model (shared per mode/case/seed)
  gradcheck  finite-difference audit of the pH rollout gradient used by shooting/selector
  train      train every method (pH-engine methods and the CT-SAC baseline)
  eval       paired evaluation over test cases, clocks and conditions
  report     seed-level summary JSON + Markdown
  verify     replay recorded transitions, run the paper-equation checks, and
             (with --sensor-root) the MuJoCo reference audit of existing checkpoints
  all        prepare, gradcheck, train, eval, report
"""
import argparse
import json
import os
from pathlib import Path
import sys
from . import config

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "prepare", "gradcheck", "train", "eval", "report", "verify", "all"))
    p.add_argument("--profile", default="experiments/full.json")
    p.add_argument("--run-root", default="runs/full")
    p.add_argument("--artifacts", default="artifacts/full")
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
    p.add_argument("--output", default="reports/full.json")
    p.add_argument("--disclose-holds", action="store_true",
                   help="Ablation: show the requested hold to policies (v3.7 interface)")
    p.add_argument("--sensor-root", help="verify: existing sensor/ runs for the MuJoCo reference audit")
    return p

def verify(args):
    """Checks that need no training: exact replays, paper equations, optional reference audit."""
    import subprocess
    from mujoco_reference.transition import validate
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for sample in sorted((root / "validation" / "reference_samples").glob("*/*_transition.json")):
        try:
            result = validate(sample)
            print(json.dumps(dict(check="transition_replay", sample=str(sample.relative_to(root)),
                                  passed=result["passed"], force_balance_inf=result["force_balance_inf"])))
        except Exception as exc:
            failures += 1
            print(json.dumps(dict(check="transition_replay", sample=str(sample.relative_to(root)),
                                  passed=False, error=str(exc))))
    code = subprocess.call([sys.executable, str(root / "tools" / "paper_checks.py")])
    failures += int(code != 0)
    if args.sensor_root:
        output = Path(args.run_root) / "mujoco_reference"
        code = subprocess.call([sys.executable, "-m", "mujoco_reference", "compare",
                                "--sensor-root", args.sensor_root, "--output", str(output)], cwd=root)
        failures += int(code != 0)
    return 1 if failures else 0

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
    if args.disclose_holds:
        profile["learning"]["hold_disclosure"] = True
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
    if args.stage == "verify":
        return verify(args)
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
    stages = ("prepare", "gradcheck", "train", "eval", "report") if args.stage == "all" else (args.stage,)
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
                    if job["method"] in config.BASELINES:
                        continue
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

