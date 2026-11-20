"""Run a small acquisition/partition check. Fit no observer, operator, or policy."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['F','J','both'],default='both')
    p.add_argument('--config',default='cases/path_speed.json')
    p.add_argument('--decisions',type=int,default=256)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',default='prechecks/calibration')
    a=p.parse_args()
    if a.decisions<16:p.error('Use at least 16 diagnostic decisions')
    os.environ.setdefault('MUJOCO_GL','egl')
    from fjbench.config import config_from
    from fjbench.calibration import acquire,partition_acquisition
    from bench.protocol import write_json,versions
    failed=False
    for case in (['F','J'] if a.case=='both' else [a.case]):
        c=config_from(a.config,case,'contact',a.seed,
            extra={'calibration_decisions':a.decisions,'batch':min(32,a.decisions)})
        replay,counters,backend=acquire(c)
        _,_,admission=partition_acquisition(c,replay,backend)
        report=dict(config=c,counters=counters,admission=admission,backend=backend,
            observer_updates=0,operator_updates=0,policy_updates=0,versions=versions())
        path=Path(a.out)/f'{case}_seed{a.seed}_acquisition.json';write_json(path,report)
        print(json.dumps(dict(case=case,admitted=admission['admitted'],decisions=replay.count,
            goal_coverage=admission['path_goal_coverage_by_task'],report=str(path))))
        failed|=not admission['admitted']
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
