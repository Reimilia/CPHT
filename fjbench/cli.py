"""Dependency-light command configuration and plan loading."""
import argparse
import json
import os
from pathlib import Path
from .config import make_config,config_from,calibration_key,run_key,METHODS


def parser(description,training=False):
    p=argparse.ArgumentParser(description=description)
    p.add_argument("--case",choices=["F","J"],default="F")
    p.add_argument("--experiment")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--config",help="JSON setting overrides")
    p.add_argument("--plan",help="Optional JSONL run plan")
    p.add_argument("--index",type=int,default=int(os.environ.get("SLURM_ARRAY_TASK_ID","0")))
    p.add_argument("--artifacts",default="artifacts/fj")
    p.add_argument("--root",default="runs")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda" if training else "cpu")
    p.add_argument("--dry-run",action="store_true")
    if training:p.add_argument("--method",choices=METHODS,default="ph_sac")
    return p


def resolve(args):
    method=getattr(args,"method",None)
    if args.plan:
        records=[s for s in Path(args.plan).read_text().splitlines() if s.strip()]
        entry=json.loads(records[args.index])
        c=entry["config"]
        if make_config(c["case"],c["experiment"],c["seed"],c)!=c:
            raise ValueError("Invalid plan configuration")
        method=entry.get("method",method)
        if entry["calibration"]!=calibration_key(c):raise ValueError("Calibration plan identity differs")
        if method and entry.get("run_id",run_key(c,method))!=run_key(c,method):raise ValueError("Run plan identity differs")
    else:c=config_from(args.config,args.case,args.experiment,args.seed)
    prepared=Path(args.artifacts).resolve()/calibration_key(c)/"prepared.pt"
    directory=Path(args.root).resolve()/run_key(c,method) if method else None
    return c,method,prepared,directory
