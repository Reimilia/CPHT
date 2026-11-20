"""Shared measured calibration replay, ordinary SAC updates, boundary recovery."""
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from bench.protocol import write_json,versions
from bench.training import atomic_torch_save
from .config import feature_kind,calibration_id
from .mechanics import NominalRobot,PhysicalEnvironment,ph_features
from .prepare import load,feature_arrays
from .learner import Learner,Replay


def normalized(packet,kind,coefficients,norm):
    vector=packet['raw'] if kind=='raw' else ph_features(packet['raw'],packet['basis'],coefficients)
    return ((vector-norm[kind+'_mean'])/norm[kind+'_scale']).astype(np.float32)


def train(s,c,method,artifact,directory,device='cpu',resume=False,pause_after_episodes=None):
    torch.manual_seed(s['seed']);torch.set_num_threads(1)
    calibration,data,norm=load(artifact);coefficients=calibration['coefficients'];kind=feature_kind(method)
    if calibration['calibration_id']!=calibration_id(s,c):raise ValueError('Training and calibration contracts differ')
    features=feature_arrays(data,coefficients)
    dimension=features[kind].shape[1]
    replay=Replay(s['capacity'],dimension,s['seed']+173)
    for i in norm['train_indices']:
        replay.add(obs=(features[kind][i]-norm[kind+'_mean'])/norm[kind+'_scale'],
            next_obs=(features['next_'+kind][i]-norm[kind+'_mean'])/norm[kind+'_scale'],
            **{key:data[key][i] for key in ('action','R','r_src','h','h_requested','next_h','done')})
    agent=Learner(dimension,s,c,method,device);nominal=NominalRobot(s)
    if nominal.xml_sha256!=calibration['nominal_xml_sha256']:raise ValueError('Robot XML differs from physical calibration')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    counters=dict(decisions=s['calibration_decisions'],policy_decisions=0,policy_physics_steps=0,updates=0,
        policy_force_sensor_solves=0,nominal_queries=0,linear_value_solves=0)
    episode=0
    if resume:
        saved=torch.load(directory/'latest.pt',map_location=device,weights_only=False)
        if saved['settings']!=s or saved['control']!=c or saved['method']!=method:raise ValueError('Resume contract changed')
        agent.load(saved['agent']);replay.restore(saved['replay']);counters=saved['counters'];episode=saved['next_episode']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        for name in ('train.jsonl','episodes.jsonl'):
            path=directory/name
            if path.exists():path.write_text(''.join(row+'\n' for row in path.read_text().splitlines() if json.loads(row)['decisions']<=counters['decisions']))
    elif (directory/'latest.pt').exists():raise FileExistsError('Run exists; use --resume or a new root')
    write_json(directory/'run.json',dict(settings=s,control=c,method=method,feature_kind=kind,
        measurement_contract=calibration['policy_measurements'],calibration=calibration,versions=versions(),
        purpose='Physical calibration feature ablation; ordinary SAC/CT-SAC targets',finite_value_solver=False))
    stop={'requested':False}
    def stopping(*_):stop['requested']=True
    signals=(signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGUSR1',signal.SIGTERM))
    previous_handlers={sig:signal.getsignal(sig) for sig in signals}
    for sig in previous_handlers:signal.signal(sig,stopping)
    started=time.perf_counter();last_save=counters['decisions'];initial_episode=episode;prior_queries=counters['nominal_queries']
    def save(path,recovery=False):
        value=dict(schema=s['schema'],settings=s,control=c,method=method,feature_kind=kind,dimension=dimension,
            calibration=calibration,normalization=norm,agent=agent.state(),counters=counters.copy())
        if recovery:value.update(replay=replay.state(),next_episode=episode,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        atomic_torch_save(value,path)
    try:
        with (directory/'train.jsonl').open('a',buffering=1) as log,(directory/'episodes.jsonl').open('a',buffering=1) as elog:
            while counters['decisions']<s['total_decisions'] and not stop['requested']:
                env=PhysicalEnvironment(c,s,1000000000+s['seed']*10000+episode,episode,nominal=nominal)
                stats={};complete=False
                try:
                    while counters['decisions']<s['total_decisions']:
                        current=normalized(env.current,kind,coefficients,norm);requested=env.hold()
                        action=agent.act(current)
                        successor,info,_=env.step(action)
                        replay.add(obs=current,next_obs=normalized(successor,kind,coefficients,norm),action=action,
                            R=info['R'],r_src=info['r_src'],h=info['h'],h_requested=requested,next_h=info['next_h'],
                            done=float(info['terminated'] or info['truncated']))
                        stats=agent.update(replay.sample(s['batch'],device))
                        counters['decisions']+=1;counters['policy_decisions']+=1;counters['updates']+=1
                        counters['policy_physics_steps']+=info['physics_steps']
                        if counters['decisions']%s['log_every']==0:
                            log.write(json.dumps(dict(counters,diagnostics=stats,wall_seconds=time.perf_counter()-started))+'\n')
                        if info['terminated'] or info['truncated']:complete=True;break
                    elog.write(json.dumps(dict(decisions=counters['decisions'],episode=episode,complete=complete,path_metrics=env.path_task.summary(False)))+'\n')
                finally:
                    counters['policy_force_sensor_solves']+=env.sensor_solves;env.close()
                counters['nominal_queries']=prior_queries+nominal.queries
                episode+=1
                if pause_after_episodes is not None and episode-initial_episode>=pause_after_episodes:stop['requested']=True
                if counters['decisions']-last_save>=s['save_every'] or stop['requested']:
                    save(directory/'latest.pt',True);last_save=counters['decisions']
                    save(directory/f"policy_{counters['decisions']:09d}.pt")
        save(directory/'latest.pt',True)
        if counters['decisions']>=s['total_decisions']:save(directory/'final.pt')
        write_json(directory/'status.json',dict(complete=counters['decisions']>=s['total_decisions'],counters=counters))
        return counters
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
