"""Separate, explicitly instrumented physical-first experiment contract."""
from pathlib import Path
import math
from bench.protocol import digest
from fjbench.config import config_from,calibration_key

METHODS=('ct_sac','cpht_sac','sac','ct_sac_ph','pure_cpht')
TERRAINS=('hills','friction','terrain','slopes','steps')


def settings(case='F',terrain='terrain',seed=0,**extra):
    s=dict(schema='physical_first_v1',case=case,terrain=terrain,seed=int(seed),
        total_decisions=1_000_000,calibration_decisions=2048,calibration_episode_steps=32,
        nominal_mass_scale=.8,nominal_stiffness_scale=1.2,nominal_damping_scale=1.4,hidden=[256,256],batch=256,
        capacity=300000,lr=3e-4,tau=.005,save_every=100000,log_every=1000)
    unknown=set(extra)-set(s)-{'pure','sensing'}
    if unknown:raise ValueError(f'Unknown physical-first settings: {unknown}')
    s.update(extra)
    if 'pure' in s:
        from pure_cpht.config import options
        s['pure']=options(s['pure'])
    if 'sensing' in s:
        from environment_belief.config import options as sensing_options
        s['sensing']=sensing_options(s['sensing'])
    if case not in ('F','J') or terrain not in TERRAINS:raise ValueError('Invalid case/terrain')
    counts=('seed','total_decisions','calibration_decisions','calibration_episode_steps','batch','capacity','save_every','log_every')
    if any(not isinstance(s[k],int) or isinstance(s[k],bool) for k in counts):raise ValueError('Counts and seed must be integers')
    if not 0<=s['seed']<3000:raise ValueError('Use a seed in [0,3000)')
    if s['calibration_decisions']<16 or s['total_decisions']<=s['calibration_decisions']:
        raise ValueError('Reserve at least 16 calibration decisions and a positive policy budget')
    if any(not math.isfinite(s[k]) or s[k]<=0 for k in ('nominal_mass_scale','nominal_stiffness_scale','nominal_damping_scale','lr','tau')):raise ValueError('Positive finite physical scales and learning rates required')
    if s['tau']>1:raise ValueError('Target averaging tau must be in (0,1]')
    if not s['hidden'] or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in s['hidden']):raise ValueError('Hidden widths must be positive integers')
    if min(s[k] for k in ('batch','capacity','calibration_episode_steps','save_every','log_every'))<1:raise ValueError('Positive counts required')
    if s['capacity']<s['calibration_decisions']:raise ValueError('Replay must retain the shared calibration data')
    return s


def control(s,path=None):
    # Keep terrain/task settings from the existing benchmark. Its observer,
    # node-bank, and value-operator preparation are unused in this pipeline.
    c=config_from(path or f"cases/path_speed_ct_{s['terrain']}.json",s['case'],'contact',s['seed'],
        extra=dict(control_task='path_speed',calibration_decisions=s['calibration_decisions'],
            total_decisions=s['total_decisions'],batch=min(s['batch'],s['calibration_decisions'])))
    if c['contact_robot_variation']:raise ValueError('The three-parameter pilot uses the same robot across contact environments')
    if 'sensing' in s:
        if len(c['contact_offsets'])!=7:raise ValueError('The sensing experiment uses seven probe locations')
        if s['sensing']['sensor_period']<c['physics_dt']-1e-12:raise ValueError('Sensor period must be at least one physics step')
    elif s.get('pure',{}).get('terrain_probes') and c['contact_sensor']!='geometry':raise ValueError('Pure CPHT uses the declared geometric probe sensor')
    return c


def calibration_id(s,c):
    contract=dict(pipeline=s['schema'],measurements='physics_rate_ports_v1',control=calibration_key(c),
        episode_steps=s['calibration_episode_steps'],mass=s['nominal_mass_scale'],stiffness=s['nominal_stiffness_scale'],damping=s['nominal_damping_scale'])
    if s.get('pure',{}).get('terrain_probes'):contract['measurements']='physics_rate_ports_terrain_v2'
    if 'sensing' in s:
        contract.update(measurements='physics_rate_environment_belief_v1',sensing=s['sensing'],
            point_labels='pure' in s and s['pure'].get('contact_supervision','point_labels')=='point_labels',
            contact_closure_data='pure' in s,contact_supervision=s.get('pure',{}).get('contact_supervision','point_labels'))
    return digest(contract)[:12]


def artifact_path(root,s,c):return Path(root)/s['case']/s['terrain']/f"seed{s['seed']}_{calibration_id(s,c)}"
def run_path(root,s,c,method):
    if method not in METHODS:raise ValueError(method)
    return Path(root)/s['case']/s['terrain']/method/f"seed{s['seed']}_{digest(dict(settings=s,control=c))[:12]}"
def feature_kind(method):return 'ph' if method in ('cpht_sac','ct_sac_ph') else 'raw'
