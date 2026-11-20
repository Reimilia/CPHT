"""Write reproducible contact-map manifests and optional profile figures. No RL fit."""
import argparse
import json
from pathlib import Path
from fjbench.config import config_from


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=["F","J","both"],default="J")
    p.add_argument("--config")
    p.add_argument("--seed",type=int,default=0,help="Training configuration seed")
    p.add_argument("--reset-seed",type=int,default=100000)
    p.add_argument("--count",type=int,default=3)
    p.add_argument("--cohort",default="iid")
    p.add_argument("--progress",type=float,default=1.,help="J curriculum progress; eval previews use 1")
    p.add_argument("--out",default="prechecks/contact/environments.jsonl")
    p.add_argument("--plot",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.count<1 or not 0<=a.progress<=1:p.error("Positive count and progress in [0,1] required")
    cases=["F","J"] if a.case=="both" else [a.case]
    configs=[config_from(a.config,case,"contact",a.seed) for case in cases]
    if "F" in cases and a.cohort!="iid":p.error("F retains a fixed map; request a J cohort")
    if a.dry_run:
        print(json.dumps(dict(stage="sample",cases=cases,count=a.count,cohort=a.cohort,
            output=a.out,expensive_work_launched=False),indent=2));return
    from fjbench.contact import sample_map,plot_map
    from bench.protocol import digest
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for c in configs:
        for i in range(a.count):
            seed=a.reset_seed+i
            environment=sample_map(c,seed,a.cohort,a.progress)
            name=f"{c['case']}_{a.cohort}_{i:03d}"
            row=dict(schema=1,name=name,config=c,config_sha256=digest(c),reset_seed=seed,cohort=a.cohort,environment=environment,
                privileged_map_for_backend_and_audit_only=True,training_updates=0)
            records.append(row)
            if a.plot:plot_map(environment,out.parent/(name+"_profile.png"))
    out.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in records))
    print(json.dumps(dict(manifest=str(out.resolve()),environments=len(records),training_updates=0)))


if __name__=="__main__":main()
