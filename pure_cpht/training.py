"""Learn H/contact models from interaction; execute bounded MPC feedback.

The training record contains physical labels and planner state. No actor,
critic, Bellman target, or SAC replay update occurs in this module.
"""
from collections import deque,defaultdict
import json
from pathlib import Path
import signal
import numpy as np
import torch
from bench.protocol import write_json,versions
from bench.training import atomic_torch_save
from physical_fj.prepare import load
from physical_fj.config import calibration_id
from physical_fj.mechanics import NominalRobot,PhysicalEnvironment
from .controller import Controller
from .learning import prepare_artifact,fit_all

FIELDS=('q','v','acceleration','y','u','constraint_port')


class Bank:
    def __init__(self,capacity,rows=()):self.rows=deque(rows,maxlen=capacity);self.capacity=capacity
    @classmethod
    def from_data(cls,capacity,data):
        contacts=defaultdict(list)
        for j,i in enumerate(data.get('contact_sample',[])):
            contacts[int(i)].append({k.removeprefix('contact_'):v[j] for k,v in data.items() if k.startswith('contact_') and k!='contact_sample'})
        rows=[]
        for i in range(len(data['q'])):
            rows.append(dict({k:data[k][i] for k in FIELDS},scan=data['physics_scan'][i],contacts=contacts[i],physics_episode=int(data['physics_episode'][i])))
        return cls(capacity,rows)
    def add(self,samples,episode):
        for sample in samples:self.rows.append(dict({k:sample[k] for k in FIELDS},scan=sample['scan'],contacts=sample['contacts'],physics_episode=int(episode)))
    def dataset(self,maximum):
        rows=list(self.rows)
        if len(rows)>maximum:rows=[rows[i] for i in np.linspace(0,len(rows)-1,maximum,dtype=int)]
        result={k:np.asarray([r[k] for r in rows]) for k in FIELDS+('physics_episode',)}
        result['physics_scan']=np.asarray([r['scan'] for r in rows]);contacts=[]
        for i,row in enumerate(rows):
            contacts.extend(dict(c,sample=i) for c in row['contacts'])
        if contacts:result.update({'contact_'+k:np.asarray([r[k] for r in contacts]) for k in contacts[0]})
        return result
    def state(self):return list(self.rows)


def train(s,c,method,artifact,directory,device='cpu',resume=False,pause_after_episodes=None):
    if method!='pure_cpht' or 'pure' not in s:raise ValueError('Use pure_cpht with cases/pure_cpht.json')
    torch.set_num_threads(1);calibration,data,norm=load(artifact)
    if calibration['calibration_id']!=calibration_id(s,c):raise ValueError('Calibration contract changed')
    prepared=prepare_artifact(artifact,s);model=prepared['model'];metrics=prepared['metrics'];neutral=prepared['neutral'];o=s['pure']
    nominal=NominalRobot(s)
    if nominal.xml_sha256!=model['geometry']['xml_sha256']:raise ValueError('Robot XML differs from calibration')
    bank=Bank.from_data(o['model_buffer'],data);gait=None;episode=0
    counters=dict(decisions=s['calibration_decisions'],policy_decisions=0,policy_physics_steps=0,
        updates=0,model_updates=0,model_refit_attempts=0,actor_updates=0,critic_updates=0,
        control_optimizations=0,predictive_substeps=0,linear_value_solves=0,policy_force_sensor_solves=0)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if resume:
        saved=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
        if saved['settings']!=s or saved['control']!=c or saved['method']!=method:raise ValueError('Resume contract changed')
        model=saved['pure_model'];metrics=saved['model_metrics'];gait=saved['gait'];counters=saved['counters'];episode=saved['next_episode']
        bank=Bank(o['model_buffer'],saved['physical_bank'])
        for name in ('train.jsonl','episodes.jsonl'):
            p=directory/name
            if p.exists():p.write_text(''.join(line+'\n' for line in p.read_text().splitlines() if json.loads(line)['decisions']<=counters['decisions']))
    elif (directory/'latest.pt').exists():raise FileExistsError('Use --resume or a new run root')
    write_json(directory/'run.json',dict(settings=s,control=c,method=method,calibration=calibration,versions=versions(),
        purpose='Experimental actor-free mechanical CPHT: positive learned H/contact closure, latent periodic potential, MPC feedback',
        observation_contract=calibration['policy_measurements'],actor=False,critic=False,bellman_updates=False))
    stopping={'requested':False}
    def stop(*_):stopping['requested']=True
    signals={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGUSR1',signal.SIGTERM))}
    for sig in signals:signal.signal(sig,stop)
    last_save=counters['decisions'];first_episode=episode
    def save(path,recovery=False):
        packet=dict(schema=s['schema'],settings=s,control=c,method=method,dimension=calibration['raw_observation_dimension'],feature_kind='raw',
            calibration=calibration,normalization=norm,pure_model=model,model_metrics=metrics,neutral=neutral,gait=gait,counters=counters.copy(),
            actor=None,critic=None,algorithm='learned pH + latent periodic potential + receding-horizon feedback')
        if recovery:packet.update(physical_bank=bank.state(),next_episode=episode)
        atomic_torch_save(packet,path)
    try:
        with (directory/'train.jsonl').open('a',buffering=1) as log,(directory/'episodes.jsonl').open('a',buffering=1) as elog:
            while counters['decisions']<s['total_decisions'] and not stopping['requested']:
                env=PhysicalEnvironment(c,s,1000000000+s['seed']*10000+episode,episode,nominal=nominal)
                controller=Controller(model,o,neutral,seed=s['seed']*100000+episode,device=device,gait=gait)
                complete=False;fit_report={}
                try:
                    while counters['decisions']<s['total_decisions']:
                        action=controller.act(env.current,env.g,env.hold(),env.base.cur_t)
                        _,info,labels=env.step(action,calibrate=True)
                        bank.add(labels['samples'],1000000+episode)
                        counters['decisions']+=1;counters['policy_decisions']+=1;counters['policy_physics_steps']+=info['physics_steps']
                        if counters['policy_decisions']%o['model_refit_every']==0:
                            counters['model_refit_attempts']+=1;dataset=bank.dataset(o['model_fit_samples'])
                            initial=set(calibration['metrics']['validation_episodes'])
                            present=np.unique(dataset['physics_episode'])
                            validation=[int(e) for e in present if int(e) in initial or (e>=1000000 and (e-1000000)%5==4)]
                            cal=dict(calibration,metrics=dict(calibration['metrics'],validation_episodes=validation))
                            try:
                                candidate,new_metrics=fit_all(nominal,cal,dataset,o,previous=model)
                                model,metrics=candidate,new_metrics;counters['model_updates']+=1;counters['updates']+=1
                                # Keep the phase and control-search state; replace the learned plant.
                                measured_q=controller.robot.t(env.current['raw'][:9])[None]
                                measured_v=controller.robot.t(env.current['raw'][9:18])[None]
                                old_energy=float(controller.robot.quantities(measured_q,measured_v)['H'][0])
                                controller.robot=controller.robot.__class__(model['geometry'],model['parameters'],device=device)
                                controller.contact=controller.contact.__class__(controller.robot,model['contact'],o)
                                new_energy=float(controller.robot.quantities(measured_q,measured_v)['H'][0])
                                controller.last_plan=-float('inf');fit_report=dict(admitted=True,metrics=metrics,
                                    model_energy_recalibration_jump=new_energy-old_energy)
                            except (ValueError,RuntimeError) as error:
                                fit_report=dict(admitted=False,reason=str(error))
                                write_json(directory/'model_refit_failure.json',dict(decisions=counters['decisions'],**fit_report))
                        if counters['decisions']%s['log_every']==0:
                            log.write(json.dumps(dict(decisions=counters['decisions'],counters=counters,
                                controller=controller.diagnostics,model_update=fit_report))+'\n')
                        if info['terminated'] or info['truncated']:complete=True;break
                    gait=controller.mean.cpu().tolist()
                    elog.write(json.dumps(dict(decisions=counters['decisions'],episode=episode,complete=complete,
                        path_metrics=env.path_task.summary(False),controller=controller.diagnostics))+'\n')
                finally:
                    counters['policy_force_sensor_solves']+=env.sensor_solves
                    counters['control_optimizations']+=controller.plans;counters['predictive_substeps']+=controller.predicted_steps
                    env.close()
                episode+=1
                if pause_after_episodes is not None and episode-first_episode>=pause_after_episodes:stopping['requested']=True
                if counters['decisions']-last_save>=s['save_every'] or stopping['requested']:
                    save(directory/'latest.pt',True);save(directory/f"policy_{counters['decisions']:09d}.pt");last_save=counters['decisions']
        save(directory/'latest.pt',True)
        if counters['decisions']>=s['total_decisions']:save(directory/'final.pt')
        write_json(directory/'status.json',dict(complete=counters['decisions']>=s['total_decisions'],counters=counters))
        return counters
    finally:
        for sig,handler in signals.items():signal.signal(sig,handler)
