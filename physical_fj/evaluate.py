"""Frozen physical-first policies on paired layouts, clocks, and terrain kinds."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from bench.protocol import write_json
from bench.learner import Actor
from fjbench.config import make_config
from fjbench.evaluation import path_family_metrics
from fjbench.reporting import plot_evaluation
from fjbench.recording import EpisodeRecorder
from .mechanics import NominalRobot,PhysicalEnvironment
from .train import normalized


def evaluate(checkpoint,cohort='iid',clock='fixed',episodes=5,device='cpu',video_episodes=0,
             terrain=None,overwrite=False,width=640,height=480,case_test=None,feedback_only=False,sensing_test=None):
    path=Path(checkpoint);saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('schema')!='physical_first_v1':raise ValueError('Use eval_fj.py for the earlier benchmark checkpoints')
    s,c=saved['settings'],saved['control'];original=c
    training_sensing=s.get('sensing')
    if sensing_test is not None:
        if training_sensing is None:raise ValueError('Sensing test overrides require a sensing-trained checkpoint')
        from environment_belief.config import options
        tested=options(dict(training_sensing,**sensing_test))
        if tested['mode']!=training_sensing['mode']:raise ValueError('Information modes require separate training; keep the checkpoint observation schema')
        s=dict(s,sensing=tested)
    if case_test and case_test!=c['case']:
        base=make_config(case_test,'contact',c['seed'],dict(control_task='path_speed',
            path_target_speed=c['path_target_speed'],contact_backend='ct_terrain'))
        updated=dict(c,case=case_test)
        for field in ('speed','friction','load'):
            for split in ('train','validation','test'):updated[field+'_'+split]=base[field+'_'+split]
        updated['ensembles']=base['ensembles'];updated['contact_curriculum']=False
        c=make_config(case_test,'contact',c['seed'],updated)
    if terrain:
        profile={'hills':'elevation','friction':'friction'}.get(terrain,'mixed')
        elevation=terrain if terrain in ('slopes','steps') else 'hills'
        c=make_config(c['case'],'contact',c['seed'],dict(c,contact_profile=profile,terrain_elevation=elevation))
    if 'sensing' in s and s['sensing']['sensor_period']<c['physics_dt']-1e-12:
        raise ValueError('Sensor period must be at least one physics step')
    pure=saved['method']=='pure_cpht'
    if feedback_only and not pure:raise ValueError('Feedback-only deployment is available for Pure CPHT checkpoints')
    variant='_feedback' if feedback_only else ''
    video_label=('Pure CPHT feedback' if feedback_only else 'Pure CPHT MPC') if pure else saved['method']
    if 'sensing' in s:video_label+=' | '+s['sensing']['mode']
    actor=None
    if not pure:
        actor=Actor(saved['dimension'],6,s['hidden']).to(device)
        actor.load_state_dict(saved['agent']['actor']);actor.eval().requires_grad_(False)
    nominal=NominalRobot(s);rows=[];digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if nominal.xml_sha256!=saved['calibration']['nominal_xml_sha256']:raise ValueError('The installed robot XML differs from calibration')
    kind=saved['feature_kind'];norm=saved['normalization'];coeff=saved['calibration']['coefficients']
    for i in range(episodes):
        env=PhysicalEnvironment(c,s,100000+i,i,cohort,clock,nominal=nominal,purpose='evaluation')
        controller=None;prediction=[];assimilation=[]
        if pure:
            from pure_cpht.controller import Controller
            controller=Controller(saved['pure_model'],s['pure'],saved['neutral'],100000+i,device,saved['gait'],feedback_only)
        recorder=None
        try:
            if i<video_episodes:
                tag=terrain or s['terrain']
                recorder=EpisodeRecorder(env,path.parent/'videos'/f"{path.stem}_{s['case']}to{c['case']}_{tag}_{cohort}_{clock}_{i}{variant}.mp4",
                    dict(width=width,height=height,fps=25,overwrite=overwrite,end_card_seconds=1.),f"{s['case']}/{tag} {video_label}")
            discount=1.;total=0.
            while True:
                before=env.current
                if pure:action=controller.act(before,env.g,env.hold(),env.base.cur_t)
                else:
                    observation=normalized(before,kind,coeff,norm)
                    with torch.no_grad():action=actor.sample(torch.as_tensor(observation,device=device).reshape(1,-1),True)[0][0].cpu().numpy()
                successor,info,_=env.step(action);total+=discount*info['R']
                if env.sensing is not None:assimilation.append(env.sensing.audit())
                if pure:prediction.append(controller.prediction_error(before,successor,action,info['h'],c['physics_dt']))
                discount*=c['gamma_reference']**(info['h']/c['reference_time'])
                if info['terminated'] or info['truncated']:break
            row=dict(episode_seed=100000+i,g=env.g,path_metrics=env.path_task.summary(),discounted_task_return=total,
                force_sensor_solves=env.sensor_solves,audit_truth=env.audit_truth())
            if pure:
                row.update(controller=controller.diagnostics,prediction_metrics={k:float(np.mean([r[k] for r in prediction])) for k in prediction[0]},
                    actor_updates=0,critic_updates=0,model_updates=0)
            if assimilation:
                from environment_belief.runtime import summarize
                row.update(assimilation_metrics=summarize(assimilation),assimilation_trace=assimilation,
                    contact_audit=info.get('contact_audit'))
            if recorder:
                recorder.capture(env.base.cur_t);row['video']=recorder.close(True,dict(row,checkpoint_sha256=digest));recorder=None
            rows.append(row)
        finally:
            if recorder:recorder.close(False)
            env.close()
    result=dict(config=c,training_control=original,settings=s,method=saved['method'],cohort=cohort,clock=clock,split='test',
        rows=rows,path_metrics=path_family_metrics(rows),checkpoint_sha256=digest,episodes=episodes,
        deployment_updates=0,calibration_frozen=True,finite_value_solver=False,measurement_mode='instrumented state and constraint-force ports')
    result.update(calibration_id=saved['calibration']['calibration_id'],calibration_coefficients=coeff,
        train_counters=saved['counters'])
    if 'sensing' in s:
        from environment_belief.config import contract
        result.update(information_contract=contract(s),training_sensing=training_sensing,
            evaluation_sensing=s['sensing'],measurement_mode='Versioned sensing experiment: '+s['sensing']['mode'],
            observer_state_updates=True,observer_parameter_updates=0)
    elif s.get('pure',{}).get('terrain_probes'):result['measurement_mode']+=' and noisy local height/slope probes'
    if pure:result.update(actor=False,critic=False,model_frozen=True,control_solves_are_learning_updates=False,
        model_metrics=saved['model_metrics'],algorithm='Pure CPHT: experimental actor-free mechanical pH feedback/MPC',
        controller_mode='saved_gait_feedback' if feedback_only else 'receding_horizon_feedback')
    if feedback_only:result.update(method='pure_cpht_feedback',trained_method='pure_cpht')
    name=f"{path.stem}_{s['case']}to{c['case']}_{terrain or s['terrain']}_{cohort}_{clock}_n{episodes}{variant}"
    if sensing_test is not None:
        from bench.protocol import digest
        name+='_sensor_test_'+digest(s['sensing'])[:8]
    write_json(path.parent/'evaluations'/(name+'.json'),result)
    plot_evaluation(result,path.parent/'evaluations'/(name+'.png'))
    if 'sensing' in s:
        from environment_belief.reporting import plot_trace
        plot_trace(rows[0]['assimilation_trace'],path.parent/'evaluations'/(name+'_assimilation.png'))
    return result
