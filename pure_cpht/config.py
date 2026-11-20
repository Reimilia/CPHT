"""A separate experimental mechanical CPHT controller contract."""
import math

DEFAULTS=dict(terrain_probes=True,horizon_steps=20,planning_dt=.01,substeps=2,
    candidates=48,elites=8,iterations=3,replan_seconds=.04,contact_iterations=12,max_contacts=6,
    feedback_iterations=16,model_refit_every=512,model_buffer=24000,model_fit_samples=12000,
    mass_ridge=1e-5,contact_ridge=.01,feedback_ridge=.01,
    joint_kp=60.,joint_kd=12.,height_kp=60.,height_kd=12.,pitch_kp=50.,pitch_kd=10.,
    speed_kd=3.,target_clearance=.55,speed_weight=4.,clearance_weight=30.,pitch_weight=8.,
    effort_weight=.003,progress_weight=3.,uncertainty_weight=.05,
    observer_decay=.04,observer_ridge=.001,
    gait_std=.22,frequency_min=1.,frequency_max=4.,gait_frequency=2.3,
    contact_margin=.01,limit_stiffness=1600.,limit_damping=20.)


def options(value=None):
    value=value or {};unknown=set(value)-set(DEFAULTS)-{'gait_seed_scale','contact_supervision'}
    if unknown:raise ValueError(f'Unknown Pure CPHT options: {sorted(unknown)}')
    out=dict(DEFAULTS,**value)
    integers=('horizon_steps','substeps','candidates','elites','iterations','contact_iterations','max_contacts',
        'feedback_iterations','model_refit_every','model_buffer','model_fit_samples')
    if any(not isinstance(out[k],int) or isinstance(out[k],bool) or out[k]<1 for k in integers):
        raise ValueError('Pure CPHT counts must be positive integers')
    if not 1<=out['elites']<out['candidates']:raise ValueError('Use fewer elites than candidates')
    if not isinstance(out['terrain_probes'],bool) or not out['terrain_probes']:
        raise ValueError('This contact predictor requires the declared local terrain probes; use the same options for a matched CT-SAC baseline')
    for k,v in out.items():
        if k=='contact_supervision':
            if v not in ('point_labels','aggregate_ports'):raise ValueError('Contact supervision must be point_labels or aggregate_ports')
            continue
        if k=='gait_seed_scale':
            if not math.isfinite(v) or not 0<=v<=2:raise ValueError('Gait seed scale must be in [0,2]')
            continue
        if k not in integers and k!='terrain_probes' and (not math.isfinite(v) or v<=0):
            raise ValueError(f'Positive finite option required: {k}')
    if not out['frequency_min']<=out['gait_frequency']<=out['frequency_max']:raise ValueError('Invalid gait frequency interval')
    return out
