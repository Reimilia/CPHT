"""Create preparation or training plans. No simulator or Torch import is needed."""
import argparse
import json
from pathlib import Path
from fjbench.config import make_config,config_from,DEFAULT_EXPERIMENTS,METHODS,ALGORITHM_SETS,select_methods,calibration_key,run_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=["F","J","both"],default="both")
    p.add_argument("--experiments",help="Comma-separated subcases; default is both for each case")
    p.add_argument("--seeds",help="Comma-separated seeds; defaults to 0..11 for a new plan")
    p.add_argument("--from-plan",help="Filter an existing training plan without changing its task/configuration")
    p.add_argument("--methods",help="Comma- or space-separated subset; overrides --algorithm-set")
    p.add_argument("--algorithm-set",choices=ALGORITHM_SETS,default="core")
    p.add_argument("--list-algorithms",action="store_true")
    p.add_argument("--path-length",type=float)
    p.add_argument("--target-speed",type=float)
    p.add_argument("--deadline",type=float)
    p.add_argument("--control-task",choices=["native_run","path_speed"])
    p.add_argument("--kind",choices=["prepare","train"],default="train")
    p.add_argument("--config",help="Common JSON overrides")
    p.add_argument("--base-config",help="Protected task/physical preset applied after --config; used by the terrain launcher")
    p.add_argument("--out")
    a=p.parse_args();rows=[]
    if a.list_algorithms:
        print(json.dumps(dict(algorithms=METHODS,presets=ALGORITHM_SETS),indent=2));return
    if not a.out:p.error("--out is required for a run plan")
    try:methods=select_methods(a.methods,a.algorithm_set)
    except ValueError as exc:p.error(str(exc))
    if a.from_plan:
        if a.config or a.base_config or any(x is not None for x in (a.path_length,a.target_speed,a.deadline,a.control_task)):
            p.error("A saved plan retains its task configuration")
        source=[json.loads(s) for s in Path(a.from_plan).read_text().splitlines() if s.strip()]
        source=[r for r in source if a.case=="both" or r["config"]["case"]==a.case]
        if a.seeds is not None:source=[r for r in source if r["config"]["seed"] in set(map(int,a.seeds.split(',')))]
        if a.methods is not None:
            missing=set(methods)-{r.get('method') for r in source}
            if missing:p.error("Selected methods are absent from the saved training plan: "+','.join(sorted(missing)))
            source=[r for r in source if r.get('method') in methods]
        if not source:p.error("No matching jobs in the saved training plan")
        if a.kind!='train':p.error("Saved-plan filtering is used for training-job evaluation/selection")
        for r in source:
            c=r['config']
            if make_config(c['case'],c['experiment'],c['seed'],c)!=c:p.error('Invalid saved configuration')
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in source))
        print(f"Reused {len(source)} selected training jobs from {a.from_plan}. No jobs were launched.");return
    extra=json.loads(Path(a.base_config).read_text()) if a.base_config else {}
    if a.control_task:extra["control_task"]=a.control_task
    if a.control_task=="native_run" and any(x is not None for x in (a.path_length,a.target_speed,a.deadline)):
        p.error("Path parameters require control_task=path_speed")
    for value,key in ((a.path_length,"path_length"),(a.target_speed,"path_target_speed"),(a.deadline,"horizon")):
        if value is not None:extra[key]=value;extra["control_task"]="path_speed"
    if a.deadline is not None:
        import math
        base=json.loads(Path(a.config).read_text()) if a.config else {}
        extra["episode_step_cap"]=int(math.ceil(a.deadline/base.get("min_hold",.002)))
    for case in (["F","J"] if a.case=="both" else [a.case]):
        for experiment in (a.experiments.split(",") if a.experiments else DEFAULT_EXPERIMENTS[case]):
            for seed in map(int,(a.seeds or ','.join(map(str,range(12)))).split(",")):
                c=config_from(a.config,case,experiment,seed,extra)
                for method in ([None] if a.kind=="prepare" else methods):
                    entry=dict(config=c,calibration=calibration_key(c),kind=a.kind)
                    if method:entry.update(method=method,run_id=run_key(c,method))
                    rows.append(entry)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print(f"Wrote {len(rows)} {a.kind} jobs to {out}. No jobs were launched.")


if __name__=="__main__":main()
