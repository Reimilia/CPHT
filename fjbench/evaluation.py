"""Paired held-out evaluation with optional physical-time videos and path metrics."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from render_fj import FrozenPolicy,backend_differences
from .data import initial_history,next_history
from .environment import Environment
from .config import is_path_task
from .checkpoints import checkpoint_path
from bench.protocol import write_json,versions


def path_family_metrics(rows):
    if not rows or 'path_metrics' not in rows[0]:return None
    keys=('success','completed_fraction','speed_rmse','speed_mae','relative_speed_rmse',
          'speed_band_fraction','fall','timeout','discounted_task_return')
    result={}
    tasks=sorted({r['g'] for r in rows})
    for key in keys:
        result['success_rate' if key=='success' else key]=float(np.mean([
            np.mean([r[key] if key=='discounted_task_return' else r['path_metrics'][key]
                     for r in rows if r['g']==g]) for g in tasks]))
    completed=[r['path_metrics'] for r in rows if r['path_metrics']['success']]
    result['completion_time_success_only']=float(np.mean([r['completion_time'] for r in completed])) if completed else None
    result['arrival_time_error_success_only']=float(np.mean([r['arrival_time_error'] for r in completed])) if completed else None
    result['speed_rmse_success_only']=float(np.mean([r['speed_rmse'] for r in completed])) if completed else None
    result['speed_band_fraction_success_only']=float(np.mean([r['speed_band_fraction'] for r in completed])) if completed else None
    return result


def evaluate(c,method,directory,checkpoint,cohort,clock,episodes,device,split='test',video_options=None,checkpoint_file=None):
    directory=Path(directory)
    path=Path(checkpoint_file) if checkpoint_file else checkpoint_path(directory,checkpoint)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['config']!=c or saved['method']!=method:raise ValueError('Checkpoint and requested job differ')
    policy=FrozenPolicy(saved,device)
    checkpoint_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[];seed_start=50000 if split=='validation' else 100000
    for i in range(episodes):
        env=Environment(c,seed_start+i,i,'validation' if split=='validation' else cohort,clock,purpose='evaluation')
        window=initial_history(env.observation,c['history'])
        raw=integral=elapsed=work=discounted_return=0.;discount=1.
        decisions=physical=forced=0;low_torso_time=0.
        totals=dict(audit_forward_solves=0,contact_pairs_checked=0,max_time_constant_error=0.,max_damping_ratio_error=0.,
            contact_point_work=0.,contact_slip_integral=0.,normal_impulse=0.,back_foot_normal_impulse=0.,
            front_foot_normal_impulse=0.,max_penetration=0.,max_effective_friction_error=0.)
        p=env.base._env.physics
        root0=float(p.data.xpos[p.model.name2id('torso','body'),0])
        recorder=None;complete=False
        try:
            if video_options and i<video_options.get('episodes',0):
                from .recording import EpisodeRecorder
                folder=Path(video_options.get('directory') or directory/'videos')
                name=f'{checkpoint}_{checkpoint_sha[:10]}_{split}_{cohort}_{clock}_seed{seed_start+i}.mp4'
                recorder=EpisodeRecorder(env,folder/name,video_options,f"{c['case']}/{c['experiment']} {method} {clock}")
            while True:
                action=policy.act(window,env.g,env.hold(),env.base.cur_t)
                observation,info=env.step(action)
                raw+=info['r_src'];integral+=info['reward_time_integral'];elapsed+=info['h']
                discounted_return+=discount*info['R'];discount*=c['gamma_reference']**(info['h']/c['reference_time'])
                work+=info['external_work'];decisions+=1;physical+=info['physics_steps']
                forced=max(forced,info['directly_forced_dofs'])
                if 'contact_audit' in info:
                    low_torso_time+=info['h']*info['contact_audit']['low_torso']
                    for key in totals:
                        value=info['contact_audit'][key]
                        totals[key]=max(totals[key],value) if key.startswith('max_') else totals[key]+value
                window=next_history(window,observation,action,info['r_src'],info['h'])
                if info['terminated'] or info['truncated']:break
            row=dict(episode_seed=seed_start+i,g=env.g,time_score=1000*integral/c['horizon'],
                discounted_task_return=discounted_return,raw_decision_return=raw,elapsed_seconds=elapsed,
                horizon_coverage=elapsed/c['horizon'],decisions=decisions,physics_steps=physical,external_work=work,
                directly_forced_dofs=forced,environment_sensor_samples=(decisions+1)*int(c['experiment']=='contact' and c['contact_sensor']!='history'),
                odometry_samples=(decisions+1)*int(is_path_task(c)),
                contact_audit=dict(totals,low_torso_time=low_torso_time,low_torso_fraction=low_torso_time/max(elapsed,1e-9),
                    net_distance=info['contact_audit']['root_x']-root0) if c['experiment']=='contact' else None,
                audit_truth=env.audit_truth())
            row['backend']=env.backend
            row['backend_differences']=backend_differences(saved['public'].get('backend'),env.backend)
            if env.path_task is not None:row['path_metrics']=env.path_task.summary()
            if recorder is not None:
                recorder.capture(env.base.cur_t)
                video=recorder.close(True,dict(row,config=c,method=method,checkpoint_sha256=checkpoint_sha))
                row['video']={k:video[k] for k in ('video','frames','fps','physical_frames','terminal_hold_frames')}
            rows.append(row);complete=True
        finally:
            if recorder is not None and not complete:recorder.close(False)
            env.close()
    by_task={str(g):float(np.mean([r['time_score'] for r in rows if r['g']==g])) for g in sorted({r['g'] for r in rows})}
    result=dict(config=c,method=method,checkpoint=checkpoint,checkpoint_sha256=checkpoint_sha,
        calibration_sha256=saved.get('calibration_sha256'),cohort=cohort,clock=clock,split=split,episodes=episodes,
        policy_mode='squashed_mean',deployment_updates=0,task_means=by_task,family_time_score=float(np.mean(list(by_task.values()))),
        rows=rows,train_counters=saved.get('counters',{}),physical_and_policy_certificates=False,
        heldout_task_operator_certificate=False,path_metrics=path_family_metrics(rows),versions=versions())
    name=f'{checkpoint}_{split}_{cohort}_{clock}_n{episodes}.json'
    write_json(directory/'evaluations'/name,result)
    if is_path_task(c):
        from .reporting import plot_evaluation
        plot_evaluation(result,directory/'evaluations'/name.replace('.json','.png'))
    return result
