"""Short read-only Pure CPHT rollout from a checkpoint or a small model JSON."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--checkpoint-file');source.add_argument('--model-json')
    p.add_argument('--seconds',type=float,default=1.);p.add_argument('--out',default='prechecks/pure_feedback.mp4')
    p.add_argument('--seed',type=int,default=100000);p.add_argument('--clock',choices=['fixed','irregular'],default='irregular')
    p.add_argument('--replan',action='store_true',help='Recompute motion proposals; default uses the saved periodic potential')
    p.add_argument('--device',default='cpu');p.add_argument('--gl',choices=['egl','osmesa'],default='egl');p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();os.environ['MUJOCO_GL']=a.gl;os.environ['PYOPENGL_PLATFORM']=a.gl
    import torch
    from bench.protocol import write_json
    from fjbench.config import make_config
    from fjbench.recording import EpisodeRecorder
    from physical_fj.mechanics import PhysicalEnvironment
    from pure_cpht.controller import Controller
    torch.set_num_threads(1)
    file=Path(a.checkpoint_file or a.model_json)
    saved=torch.load(file,map_location='cpu',weights_only=False) if a.checkpoint_file else json.loads(file.read_text())
    if 'pure_model' not in saved:p.error('Use a Pure CPHT checkpoint/model')
    s=saved['settings'];old=saved['control'];c=make_config(old['case'],old['experiment'],old['seed'],old)
    if not 0<a.seconds<=c['horizon']:p.error('Clip duration must be positive and no longer than the saved task deadline')
    env=PhysicalEnvironment(c,s,a.seed,clock=a.clock,purpose='evaluation')
    controller=Controller(saved['pure_model'],s['pure'],saved['neutral'],a.seed,a.device,saved.get('gait'),feedback_only=not a.replan)
    recorder=None;records=[]
    try:
        recorder=EpisodeRecorder(env,Path(a.out),dict(width=640,height=480,fps=25,seconds=a.seconds,end_card_seconds=0.,overwrite=a.overwrite),
            'Pure CPHT | short '+('MPC' if a.replan else 'feedback')+' precheck')
        while env.base.cur_t<a.seconds:
            u=controller.act(env.current,env.g,env.hold(),env.base.cur_t)
            _,info,_=env.step(u);records.append(dict(time=env.base.cur_t,**controller.diagnostics))
            if info['terminated'] or info['truncated']:break
        result=dict(scope='Short precheck; the path task can remain unfinished',source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            seed=a.seed,clock=a.clock,physical_seconds=env.base.cur_t,feedback_only=not a.replan,
            model_updates=0,actor_updates=0,critic_updates=0,settings=s,control=c,path_metrics=env.path_task.summary(),
            controller=controller.diagnostics,records=records)
        # Successful recording is independent of completion of the path task.
        result['video']=recorder.close(True,dict(result));recorder=None
        write_json(Path(a.out).with_suffix('.summary.json'),result)
        print(json.dumps(dict(video=a.out,physical_seconds=result['physical_seconds'],path_metrics=result['path_metrics'])))
    finally:
        if recorder:recorder.close(False)
        env.close()


if __name__=='__main__':main()
