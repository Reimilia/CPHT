"""Evaluate a physical-first checkpoint with optional new terrain laws and MP4s."""
import argparse
import json
import os
from physical_fj.config import TERRAINS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-file',required=True)
    p.add_argument('--terrain-test',choices=TERRAINS,help='Change only the physical terrain law; freeze calibration and policy')
    p.add_argument('--case-test',choices=['F','J'],help='Use fixed or fresh courses while retaining the trained policy and calibration')
    p.add_argument('--feedback-only',action='store_true',help='Pure CPHT: reuse the saved periodic potential with feedback and no motion replanning')
    p.add_argument('--sensing-test',help='JSON sensing/environment test overrides; retains trained information mode and weights')
    p.add_argument('--cohorts',default='iid');p.add_argument('--clocks',default='fixed,irregular')
    p.add_argument('--episodes',type=int,default=5);p.add_argument('--video-episodes',type=int,default=1)
    p.add_argument('--device',default='cpu');p.add_argument('--gl',choices=['egl','osmesa'],default='egl')
    p.add_argument('--overwrite-videos',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.episodes<1 or not 0<=a.video_episodes<=a.episodes:p.error('Invalid episode count')
    clocks=a.clocks.split(',');cohorts=a.cohorts.split(',')
    if any(x not in ('fixed','irregular') for x in clocks):p.error('Invalid clock')
    if any(x not in ('iid','layout','task') for x in cohorts):p.error('Use iid, layout, or task')
    if a.dry_run:
        print(json.dumps(dict(stage='physical_first_evaluate',checkpoint=a.checkpoint_file,terrain_test=a.terrain_test,case_test=a.case_test,
            episodes=a.episodes,clocks=clocks,cohorts=cohorts,feedback_only=a.feedback_only,sensing_test=a.sensing_test,expensive_work_launched=False),indent=2));return
    os.environ['MUJOCO_GL']=a.gl;os.environ['PYOPENGL_PLATFORM']=a.gl
    import torch
    torch.set_num_threads(1)
    from physical_fj.evaluate import evaluate
    overrides=None
    if a.sensing_test:
        from pathlib import Path
        overrides=json.loads(Path(a.sensing_test).read_text())
    for cohort in cohorts:
        for clock in clocks:
            report=evaluate(a.checkpoint_file,cohort,clock,a.episodes,a.device,a.video_episodes,a.terrain_test,a.overwrite_videos,case_test=a.case_test,feedback_only=a.feedback_only,sensing_test=overrides)
            print(json.dumps(dict(cohort=cohort,clock=clock,path_metrics=report['path_metrics'])))


if __name__=='__main__':main()
