"""Write one JSON object per run. Creating a plan never starts training."""
import argparse
import json
from pathlib import Path
from bench.protocol import make_job, TASKS, RELEASED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=["released", "matched"], required=True)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--algos", help="Comma-separated methods")
    p.add_argument("--seeds", default=",".join(str(i) for i in range(12)))
    p.add_argument("--rank", default="top", choices=["top", "second", "third"])
    p.add_argument("--clock", default="irregular", choices=["irregular", "regular"])
    p.add_argument("--settings", help="JSON file with common matched settings")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    methods = a.algos.split(",") if a.algos else (RELEASED if a.track == "released" else
              ["sac_smdp", "ct_sac_corrected", "ct_picard", "tpht", "collocation_direct"])
    settings = json.loads(Path(a.settings).read_text()) if a.settings else None
    jobs = [make_job(a.track, task, algo, int(seed), a.rank, a.clock, settings)
            for task in a.tasks.split(",") for algo in methods for seed in a.seeds.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs))
    print(f"Wrote {len(jobs)} jobs to {out}. No training started.")


if __name__ == "__main__":
    main()
