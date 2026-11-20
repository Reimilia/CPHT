"""Strict pairing, seed-level summaries, and explicit failure/censoring diagnostics."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
from bench.protocol import digest, write_json

def evaluation_signature(profile):
    e = profile["evaluation"]
    return digest(dict(episodes=e["episodes"], max_decisions=e["max_decisions"]))[:8]

def failure_flags(row, profile):
    flags = []
    path = row["path_metrics"]
    if row["error"]:
        flags.append("execution_error")
    if row["budget_censored"]:
        flags.append("evaluation_budget_censored")
    if path.get("fall"):
        flags.append("fall")
    if path.get("timeout"):
        flags.append("task_deadline_timeout")
    if row["physical_seconds"] >= 0.5 and path.get("completed_fraction", 0) < 0.01:
        flags.append("little_forward_progress")
    if not row["model_unchanged"]:
        flags.append("unexpected_model_mutation")
    for name, count in row["failure_counts"].items():
        if count:
            flags.append(name)
    trace = row["controller_trace"]
    if trace and mean(x.get("action_saturation", 0) for x in trace) > 0.5:
        flags.append("frequent_motor_saturation")
    predictions = row["predictions"]
    if predictions and any(x["predicted_invalid"] for x in predictions):
        flags.append("invalid_predicted_trajectory")
    if predictions and max(x["scan_extrapolation_fraction"] for x in predictions) > 0.25:
        flags.append("prediction_outside_scan")
    if predictions and max(x["velocity_prediction_l2"] for x in predictions) > profile["optimization"]["prediction_error_limit"]:
        if any(x.get("predicted_search_gain", 0) > 1e-5 for x in trace):
            flags.append("model_mismatch_after_predicted_improvement")
    assimilation = row.get("assimilation", {})
    if assimilation and assimilation.get("informative_sliding_samples", 0) == 0:
        flags.append("no_informative_friction_evidence")
    return sorted(set(flags))

def estimate(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not finite:
        return dict(mean=None, independent_training_seeds=0, standard_error=None)
    return dict(mean=mean(finite), independent_training_seeds=len(finite),
                standard_error=stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else None)

def seed_summary(report):
    rows = report["rows"]
    completed = [r for r in rows if not r["budget_censored"]]
    predictions = [p for r in rows for p in r["predictions"]]
    decisions = [d for r in rows for d in r["controller_trace"]]
    return dict(
        success_rate=mean(float(r["path_metrics"].get("success", False) and not r["error"]) for r in completed) if completed else None,
        evaluated_episodes=len(rows), uncensored_episodes=len(completed),
        speed_rmse=mean(r["path_metrics"]["speed_rmse"] for r in rows),
        completed_fraction=mean(r["path_metrics"]["completed_fraction"] for r in rows),
        physical_seconds=mean(r["physical_seconds"] for r in rows),
        discounted_return=mean(r["discounted_task_return"] for r in rows),
        velocity_prediction_l2=mean(p["velocity_prediction_l2"] for p in predictions) if predictions else None,
        decision_seconds=mean(d.get("decision_seconds", 0) for d in decisions) if decisions else None,
        training_policy_seconds=report["train_counters"]["physical_seconds"],
        training_physics_steps=report["train_counters"]["physics_steps"],
        model_rollout_substeps=sum(r["predictive_substeps"] for r in rows),
        flags=dict(Counter(flag for r in rows for flag in r["failure_flags"])),
    )

def validate_pairing(reports):
    groups = defaultdict(list)
    for r in reports:
        j = r["job"]
        key = (j["mode"], j["case"], r["test_case"], j["horizon"], j["seed"], r["clock"], r["condition"])
        groups[key].append(r)
    for key, batch in groups.items():
        methods = [r["job"]["method"] for r in batch]
        if len(methods) != len(set(methods)):
            raise ValueError(f"Duplicate method report in group {key}")
        for field in ("calibration_id", "model_sha256", "evaluation_sensing", "task", "information_contract"):
            if len({digest(r[field]) for r in batch}) != 1:
                raise ValueError(f"Unmatched {field} in group {key}")
        signatures = {tuple((x["episode_seed"], x["target_speed"],
                             x["audit_truth"].get("contact_map_sha256")) for x in r["rows"]) for r in batch}
        if len(signatures) != 1:
            raise ValueError(f"Unpaired evaluation courses/tasks in group {key}")
    return len(groups)

def collect(profile, run_root, artifact_root, output):
    from .config import jobs, paths, evaluation_jobs
    reports, missing = [], []
    for job in jobs(profile):
        _, _, _, run = paths(profile, job, run_root, artifact_root)
        for case, clock, condition in dict.fromkeys(evaluation_jobs(profile, job)):
            file = run / "evaluations" / f"{job['case']}to{case}_{clock}_{condition}_{evaluation_signature(profile)}.json"
            if not file.exists():
                missing.append(str(file))
                continue
            r = json.loads(file.read_text())
            if r.get("diagnostic_version") != "relative_contact_residual_v1":
                raise ValueError(f"Stale diagnostic schema; reevaluate with --overwrite: {file}")
            if r["job"] != job:
                raise ValueError(f"Report job mismatch: {file}")
            reports.append(r)
    paired = validate_pairing(reports)
    group = defaultdict(list)
    for r in reports:
        j = r["job"]
        key = "/".join(map(str, (j["mode"], j["case"] + "to" + r["test_case"],
                                f"h{j['horizon']:g}", r["clock"], r["condition"], j["method"])))
        group[key].append((j["seed"], seed_summary(r)))
    aggregates = {}
    for key, values in sorted(group.items()):
        if len({seed for seed, _ in values}) != len(values):
            raise ValueError("Duplicate training-seed aggregate")
        aggregate = {name: estimate([v[name] for _, v in values])
                     for name in ("success_rate", "speed_rmse", "completed_fraction", "physical_seconds",
                                  "discounted_return", "velocity_prediction_l2", "decision_seconds")}
        flags = Counter()
        for _, value in values:
            flags.update(value["flags"])
        aggregates[key] = dict(metrics=aggregate, flags=dict(flags),
                               seeds={str(seed): value for seed, value in values})
    expected_seeds = set(profile["seeds"])
    incomplete_seed_groups = [key for key, values in group.items()
                              if {seed for seed, _ in values} != expected_seeds]
    result = dict(schema=profile["schema"], smoke_only=profile["smoke_only"],
                  comparisons=aggregates, report_count=len(reports), paired_groups=paired,
                  missing_count=len(missing), missing_reports=missing,
                  incomplete_seed_groups=incomplete_seed_groups,
                  full_protocol_executed=bool(reports) and not profile["smoke_only"] and not missing
                    and not incomplete_seed_groups and all(r["complete_course_evaluated"] for r in reports),
                  statistical_unit="independent training seed; episodes averaged within seed",
                  gradient_method="piecewise contact/box-QP derivative through calibrated pH mechanics",
                  interpretation="Flags include inference limits and numerical diagnostics, separate from task failures")
    write_json(output, result)
    destination = Path(output).with_suffix(".md")
    text = ["# CPHT experiment report", "",
            f"Reports: {len(reports)}. Missing: {len(missing)}. Smoke only: {profile['smoke_only']}.", "",
            "Task success is omitted when every episode is budget-censored.", "",
            "| Condition/method | Completion | Speed RMSE | Mean decision seconds | Flags |",
            "| --- | --- | --- | --- | --- |"]
    for key, aggregate in aggregates.items():
        metrics = aggregate["metrics"]
        fmt = lambda x: "n/a" if x["mean"] is None else f"{x['mean']:.4g}"
        text.append(f"| {key} | {fmt(metrics['success_rate'])} | {fmt(metrics['speed_rmse'])} | "
                    f"{fmt(metrics['decision_seconds'])} | {', '.join(aggregate['flags']) or '-'} |")
    text += ["", "No completed-course or generalization claim follows from a smoke/censored run.", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(text))
    return result

