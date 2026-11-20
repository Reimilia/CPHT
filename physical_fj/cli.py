"""CLI common to the small physical-first pipeline."""
import argparse
import json
from pathlib import Path
from .config import settings,control,TERRAINS,METHODS,artifact_path,run_path


def parser(description,method=False):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--case',choices=['F','J'],default='F');p.add_argument('--terrain',choices=TERRAINS,default='terrain')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--options',help='JSON physical-first learning/budget options')
    p.add_argument('--config',help='JSON terrain/task overrides')
    p.add_argument('--information',choices=['state','state_belief','sensed_belief'],help='Opt in to the versioned sensing experiment')
    p.add_argument('--total-decisions',type=int);p.add_argument('--calibration-decisions',type=int)
    p.add_argument('--artifacts',default='artifacts/physical_first');p.add_argument('--root',default='runs/physical_first')
    p.add_argument('--device',default='cpu');p.add_argument('--dry-run',action='store_true')
    if method:p.add_argument('--method',choices=METHODS,default='cpht_sac')
    return p


def resolve(a):
    options=json.loads(Path(a.options).read_text()) if a.options else {}
    if getattr(a,'information',None) is not None:
        options['sensing']=dict(options.get('sensing',{}),mode=a.information)
    if set(options)&{'case','terrain','seed'}:raise ValueError('Declare case, terrain, and seed through CLI arguments')
    for key in ('total_decisions','calibration_decisions'):
        if getattr(a,key,None) is not None:options[key]=getattr(a,key)
    s=settings(a.case,a.terrain,a.seed,**options);c=control(s,a.config)
    return s,c,artifact_path(a.artifacts,s,c),run_path(a.root,s,c,getattr(a,'method','cpht_sac'))
