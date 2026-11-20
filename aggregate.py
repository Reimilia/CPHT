"""Aggregate independent training seeds. Missing evaluations cause an error."""
import argparse
import json
from pathlib import Path
import numpy as np
from bench.protocol import read_job, write_json


def bootstrap_mean(values, seed=71, samples=5000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(samples, len(values)))].mean(1)
    return list(map(float, np.quantile(draws, [.025, .975])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plans", nargs="+", required=True)
    p.add_argument("--root", default="runs")
    p.add_argument("--protocol", choices=["paper", "common"], default="common")
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--clocks", default="irregular,regular")
    p.add_argument("--contexts", default="nominal")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--metric", choices=["time_score", "raw_decision_return"], default="time_score")
    p.add_argument("--reference", default="ct_sac_corrected", help="Method for paired differences")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    groups, missing, costs, seen = {}, [], {}, set()
    mode = "stochastic" if a.stochastic else "deterministic"
    for plan in a.plans:
        count = len([s for s in Path(plan).read_text().splitlines() if s.strip()])
        for index in range(count):
            job = read_job(plan, index)
            directory = Path(a.root) / job["run_id"]
            if job["run_id"] in seen:
                raise ValueError("A run occurs more than once across plans")
            seen.add(job["run_id"])
            for clock in a.clocks.split(","):
                for context in a.contexts.split(","):
                    name = f"{a.protocol}_{a.checkpoint}_{a.split}_{mode}_{clock}_{context}_n{a.episodes}.json"
                    path = directory / "evaluations" / name
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    value = json.loads(path.read_text())
                    if value["job"] != job or len(value["rows"]) != a.episodes:
                        raise ValueError(f"Evaluation metadata mismatch: {path}")
                    expected = (50_000 if a.split == "validation" else 100_000)
                    if [r["episode_seed"] for r in value["rows"]] != list(range(expected, expected + a.episodes)):
                        raise ValueError("Evaluation seeds differ across methods")
                    key = (job["task"], clock, context, job["algo"])
                    seed_values = groups.setdefault(key, {})
                    if job["seed"] in seed_values:
                        raise ValueError("Multiple configurations for one task/method/seed. Freeze the selected plan first.")
                    seed_values[job["seed"]] = float(np.mean([r[a.metric] for r in value["rows"]]))
                    costs[job["run_id"]] = value["train_counters"]
    if missing and not a.allow_incomplete:
        raise SystemExit(f"Missing {len(missing)} evaluations. Example: {missing[0]}")
    reports = []
    for key, values in sorted(groups.items()):
        task, clock, context, algo = key
        x = np.array(list(values.values()))
        k = len(x) // 4
        item = dict(task=task, clock=clock, context=context, algo=algo,
                    training_seed_count=len(x), training_seeds=sorted(values),
                    mean=float(x.mean()), mean_ci95=bootstrap_mean(x),
                    interquartile_mean=float(np.sort(x)[k:len(x)-k].mean()))
        reference = groups.get((task, clock, context, a.reference))
        if reference is not None and algo != a.reference:
            if set(reference) != set(values) and not a.allow_incomplete:
                raise ValueError("Paired comparison requires the same training seeds")
            paired = sorted(set(reference) & set(values))
            if paired:
                differences = [values[s] - reference[s] for s in paired]
                item["paired_difference"] = dict(reference=a.reference, seeds=paired,
                            mean=float(np.mean(differences)), ci95=bootstrap_mean(differences))
        reports.append(item)
    result = dict(protocol=a.protocol, checkpoint=a.checkpoint, split=a.split, metric=a.metric,
                  aggregation_unit="independent training seed; episode returns are averaged first",
                  incomplete=bool(missing), missing=missing, results=reports, cost_by_run=costs)
    write_json(a.out, result)
    print(f"Wrote {len(reports)} result groups to {a.out}. Missing cases: {len(missing)}")


if __name__ == "__main__":
    main()
