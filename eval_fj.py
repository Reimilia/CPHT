"""Evaluate selected F/J checkpoints with task metrics, plots, and optional videos."""
import json
import os
from pathlib import Path
from fjbench.cli import parser,resolve


def main():
    p=parser(__doc__,True);p.set_defaults(device='cpu')
    p.add_argument('--cohorts',default='iid')
    p.add_argument('--clocks',default='irregular,fixed')
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--checkpoint',default='final')
    p.add_argument('--checkpoint-file',help='Load a self-contained .pt directly, including older v2/v3 checkpoints')
    p.add_argument('--video-episodes',type=int,default=0,help='Record the first N evaluated episodes from the same rollout')
    p.add_argument('--video-dir')
    p.add_argument('--video-seconds',type=float,help='Limit video length while completing numerical evaluation')
    p.add_argument('--end-card-seconds',type=float,default=1.,help='Clearly labelled terminal-state display hold; zero disables it')
    p.add_argument('--fps',type=int,default=25)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--camera',default='side')
    p.add_argument('--gl',choices=['egl','osmesa'],default='egl')
    p.add_argument('--overwrite-videos',action='store_true')
    a=p.parse_args();cohorts=a.cohorts.split(',');clocks=a.clocks.split(',')
    if a.checkpoint_file and (a.config or a.plan):p.error('Checkpoint-file mode uses the saved task and observations; omit --config/--plan')
    if a.episodes<1 or a.video_episodes<0 or a.video_episodes>a.episodes:p.error('Invalid episode/video count')
    if any(x not in ('irregular','fixed') for x in clocks):p.error('Invalid clock')
    if not 1<=a.fps<=120 or min(a.width,a.height)<128 or a.width%2 or a.height%2:p.error('Use 1..120 fps and even dimensions >=128')
    if a.video_seconds is not None and a.video_seconds<=0:p.error('Positive clip length required')
    if not 0<=a.end_card_seconds<=5:p.error('End-card duration must lie in [0,5] seconds')
    if a.split=='validation' and cohorts!=['iid']:p.error('Validation uses its disjoint bank with cohort label iid')
    options=dict(episodes=a.video_episodes,directory=a.video_dir,seconds=a.video_seconds,
        fps=a.fps,width=a.width,height=a.height,camera=a.camera,overwrite=a.overwrite_videos,end_card_seconds=a.end_card_seconds)
    if a.checkpoint_file and a.dry_run:
        print(json.dumps(dict(stage='evaluate',checkpoint_file=a.checkpoint_file,cohorts=cohorts,clocks=clocks,
            episodes=a.episodes,videos=options,expensive_work_launched=False),indent=2));return
    os.environ['MUJOCO_GL']=a.gl;os.environ['PYOPENGL_PLATFORM']=a.gl
    if a.checkpoint_file:
        import torch
        saved=torch.load(a.checkpoint_file,map_location='cpu',weights_only=False)
        c,method=saved['config'],saved['method'];directory=Path(a.checkpoint_file).resolve().parent
        a.checkpoint=Path(a.checkpoint_file).stem
    else:c,method,_,directory=resolve(a)
    allowed=('iid','regime','task','wind','joint')
    if c['experiment']=='contact':allowed+=('layout','terrain','friction','material','contact')
    if c.get('contact_backend')=='ct_terrain':allowed=('iid','layout','task')
    if any(x not in allowed for x in cohorts):p.error('Unknown cohort')
    if c['case']=='F' and cohorts!=['iid']:p.error('F retains one physical family; use J for transfer cohorts')
    if c['experiment']!='wind' and 'wind' in cohorts:p.error('Wind-law transfer belongs to J/wind')
    if c['experiment']=='contact' and not c['contact_robot_variation'] and 'regime' in cohorts:p.error('Enable contact_robot_variation for regime transfer')
    if a.dry_run:
        print(json.dumps(dict(stage='evaluate',case=c['case'],experiment=c['experiment'],method=method,
            control_task=c.get('control_task','native_run'),run=str(directory),cohorts=cohorts,clocks=clocks,
            episodes=a.episodes,videos=options,deployment_updates=0,expensive_work_launched=False),indent=2));return
    import torch
    torch.set_num_threads(1)
    from fjbench.evaluation import evaluate
    for cohort in cohorts:
        for clock in clocks:
            result=evaluate(c,method,directory,a.checkpoint,cohort,clock,a.episodes,a.device,a.split,
                            video_options=options,checkpoint_file=a.checkpoint_file)
            print(json.dumps(dict(cohort=cohort,clock=clock,family_time_score=result['family_time_score'],
                path_metrics=result['path_metrics'],videos=sum('video' in r for r in result['rows']))))


if __name__=='__main__':main()
