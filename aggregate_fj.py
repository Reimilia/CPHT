"""Aggregate independent training seeds for a declared F/J evaluation cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from aggregate import bootstrap_mean
from bench.protocol import write_json
from fjbench.config import run_key,make_config,calibration_key
from fjbench.checkpoints import checkpoint_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True);p.add_argument("--root",default="runs")
    p.add_argument("--cohort",default="iid");p.add_argument("--clock",default="irregular")
    p.add_argument("--split",choices=["validation","test"],default="test")
    p.add_argument("--episodes",type=int,default=20);p.add_argument("--checkpoint",default="final")
    p.add_argument("--reference",default="interval_euler");p.add_argument("--out",required=True)
    a=p.parse_args();groups={};missing=[];costs={};contact_groups={};path_groups={};contracts={}
    for line in Path(a.plan).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);c=row["config"];method=row["method"]
        if make_config(c["case"],c["experiment"],c["seed"],c)!=c:raise ValueError("Invalid plan")
        family=(c["case"],c["experiment"])
        contract=(calibration_key(dict(c,seed=0)),c["total_decisions"])
        if family in contracts and contracts[family]!=contract:raise ValueError("Task, shared information contract, or interaction budget differs across compared jobs")
        contracts[family]=contract
        directory=Path(a.root)/run_key(c,method)
        path=directory/"evaluations"/f"{a.checkpoint}_{a.split}_{a.cohort}_{a.clock}_n{a.episodes}.json"
        if not path.exists():missing.append(str(path));continue
        value=json.loads(path.read_text())
        expected=range(50000 if a.split=="validation" else 100000,
                       (50000 if a.split=="validation" else 100000)+a.episodes)
        if value["config"]!=c or value["method"]!=method or [r["episode_seed"] for r in value["rows"]]!=list(expected):
            raise ValueError("Evaluation information or paired seeds differ")
        checkpoint=checkpoint_path(directory,a.checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=value["checkpoint_sha256"]:
            raise ValueError("Evaluation is stale for the current checkpoint")
        key=(c["case"],c["experiment"],method)
        values=groups.setdefault(key,{})
        if c["seed"] in values:raise ValueError("Duplicate method/seed configuration; freeze a selected plan")
        values[c["seed"]]=value["path_metrics"]["success_rate"] if value.get("path_metrics") else value["family_time_score"]
        if value.get("path_metrics"):
            for metric,number in value["path_metrics"].items():
                path_groups.setdefault(key,{}).setdefault(metric,{})[c["seed"]]=number
        if c["experiment"]=="contact":
            diagnostics=contact_groups.setdefault(key,{})
            metrics=value["rows"][0]["contact_audit"].keys()
            for metric in metrics:
                per_task=[np.mean([r["contact_audit"][metric] for r in value["rows"] if r["g"]==g])
                          for g in sorted({r["g"] for r in value["rows"]})]
                diagnostics.setdefault(metric,{})[c["seed"]]=float(np.mean(per_task))
        costs[row["run_id"]]=value["train_counters"]
    if missing:raise SystemExit(f"Missing {len(missing)} evaluations. Example: {missing[0]}")
    reports=[]
    for key,values in sorted(groups.items()):
        x=np.array(list(values.values()));k=len(x)//4
        result=dict(case=key[0],experiment=key[1],method=key[2],seeds=sorted(values),
                    training_seed_count=len(values),mean=float(x.mean()),mean_ci95=bootstrap_mean(x),iqm=float(np.sort(x)[k:len(x)-k].mean()))
        if key in contact_groups:
            result["contact_metrics"]={metric:dict(mean=float(np.mean(list(per_seed.values()))),
                mean_ci95=bootstrap_mean(list(per_seed.values())),per_training_seed=per_seed)
                for metric,per_seed in contact_groups[key].items()}
        if key in path_groups:
            result["primary_metric"]="completion_rate"
            result["path_metrics"]={}
            for metric,per_seed in path_groups[key].items():
                valid={seed:number for seed,number in per_seed.items() if number is not None}
                result["path_metrics"][metric]=dict(mean=float(np.mean(list(valid.values()))) if valid else None,
                    mean_ci95=bootstrap_mean(list(valid.values())) if valid else None,
                    defined_training_seeds=sorted(valid),per_training_seed=per_seed)
        ref=groups.get((key[0],key[1],a.reference))
        if ref is not None and key[2]!=a.reference:
            if set(ref)!=set(values):raise ValueError("Paired methods have different training seeds")
            delta=[values[s]-ref[s] for s in sorted(values)]
            result["paired_difference"]=dict(reference=a.reference,mean=float(np.mean(delta)),ci95=bootstrap_mean(delta))
        reports.append(result)
    write_json(a.out,dict(cohort=a.cohort,clock=a.clock,split=a.split,checkpoint=a.checkpoint,
        aggregation="average each task, then family, then independent training seeds",results=reports,costs=costs))
    print(f"Wrote {a.out}")


if __name__=="__main__":main()
