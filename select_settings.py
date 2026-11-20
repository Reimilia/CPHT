"""Select matched settings from validation files and freeze a disjoint test plan."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics
from bench.protocol import digest, make_job, read_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plans", nargs="+", required=True)
    p.add_argument("--root", default="tuning_runs")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--test-seeds", default=",".join(str(i) for i in range(12)))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    tuning_seeds = set()
    for plan in a.plans:
        for index, _ in enumerate(s for s in Path(plan).read_text().splitlines() if s.strip()):
            j = read_job(plan, index)
            if j["track"] != "matched":
                raise ValueError("Released top/second/third ranks retain their published provenance")
            path = Path(a.root) / j["run_id"] / "evaluations" / f"common_final_validation_deterministic_irregular_nominal_n{a.episodes}.json"
            v = json.loads(path.read_text())
            if v["split"] != "validation" or v["job"] != j or len(v["rows"]) != a.episodes:
                raise ValueError("Selection requires the complete validation split")
            if [r["episode_seed"] for r in v["rows"]] != list(range(50_000, 50_000 + a.episodes)):
                raise ValueError("Validation episode seeds differ")
            groups[(j["task"], j["algo"])][digest(j["settings"])].append((j, statistics.mean(r["time_score"] for r in v["rows"])))
            tuning_seeds.add(j["seed"])
    test_seeds = [int(s) for s in a.test_seeds.split(",")]
    if tuning_seeds & set(test_seeds):
        raise ValueError("Tuning and test training seeds must be disjoint")
    candidate_counts, seed_sets = set(), set()
    jobs = []
    for (task, algo), candidates in sorted(groups.items()):
        candidate_counts.add(len(candidates))
        for rows in candidates.values():
            seeds = [j["seed"] for j, _ in rows]
            if len(seeds) != len(set(seeds)):
                raise ValueError("Duplicate tuning job")
            seed_sets.add(tuple(sorted(seeds)))
        selected = max(candidates.values(), key=lambda rows: statistics.mean(v for _, v in rows))[0][0]
        jobs.extend(make_job("matched", task, algo, seed, selected["rank"], selected["clock"], selected["settings"]) for seed in test_seeds)
    if len(candidate_counts) != 1 or len(seed_sets) != 1:
        raise ValueError("Methods must use equal candidate counts and identical tuning seeds")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs))
    print(f"Froze {len(jobs)} test runs using validation scores only: {out}")


if __name__ == "__main__":
    main()
