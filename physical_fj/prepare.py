"""Positive physical parameters, physics-rate labels, and episode holdout."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from bench.protocol import write_json,versions
from .mechanics import NominalRobot,PhysicalEnvironment,ph_features
from .config import calibration_id

BASIS=('p0','T0','Vg0','Vs0','port_power','dissipation0')
MEASUREMENT_SCHEMA='physics_rate_ports_v1'


def measurement_schema(s):
    if 'sensing' in s:return 'physics_rate_environment_belief_v1'
    return 'physics_rate_ports_terrain_v2' if s.get('pure',{}).get('terrain_probes') else MEASUREMENT_SCHEMA


def feature_arrays(data,coefficients):
    result={}
    for prefix in ('','next_'):
        raw=data[prefix+'raw'];basis={k:data[prefix+k] for k in BASIS}
        result[prefix+'raw']=raw
        result[prefix+'ph']=np.stack([ph_features(raw[i],{k:v[i] for k,v in basis.items()},coefficients) for i in range(len(raw))])
    return result


def fit(data):
    episodes=np.unique(data['episode'])
    if len(episodes)<2:raise ValueError('Physical calibration needs two disjoint episodes')
    heldout=set(episodes[4::5].tolist() or episodes[-1:].tolist())
    val=np.isin(data['episode'],list(heldout));train=~val
    physics_val=np.isin(data['physics_episode'],list(heldout));physics_train=~physics_val
    x=data['X'][physics_train].reshape(-1,3);y=data['y'][physics_train].reshape(-1)
    scale=np.sqrt(np.mean(x*x,0)).clip(1e-12)
    singular=np.linalg.svd(x/scale,compute_uv=False)
    if len(singular)!=3 or singular[-1]<1e-7*singular[0]:
        raise ValueError(f'Mass/stiffness/damping calibration is rank deficient; increase excitation or episode duration. Singular values: {singular.tolist()}')
    fitted=lsq_linear(x/scale,y,bounds=(.05*scale,20.*scale),tol=1e-12)
    if not fitted.success:raise RuntimeError(f'Physical calibration failed: {fitted.message}')
    coefficients=fitted.x/scale
    if not np.isfinite(coefficients).all():raise FloatingPointError('Non-finite physical calibration')
    reports={}
    for name,mask in (('train',physics_train),('validation',physics_val)):
        predicted=np.einsum('nij,j->ni',data['X'][mask],coefficients)
        residual=predicted-data['y'][mask];before=data['X'][mask].sum(-1)-data['y'][mask]
        power=(residual*data['v'][mask]).sum(1)
        reports[name]=dict(samples=int(mask.sum()),force_rmse=float(np.sqrt(np.mean(residual**2))),
            relative_force_rmse=float(np.sqrt(np.mean(residual**2))/max(1e-8,np.sqrt(np.mean(data['y'][mask]**2)))),
            initial_force_rmse=float(np.sqrt(np.mean(before**2))),power_balance_rmse=float(np.sqrt(np.mean(power**2))))
    features=feature_arrays(data,coefficients)
    normalization={}
    for kind in ('raw','ph'):
        normalization[kind+'_mean']=features[kind][train].mean(0)
        normalization[kind+'_scale']=features[kind][train].std(0).clip(.05)
        normalization[kind+'_mean'][27:31]=0.;normalization[kind+'_scale'][27:31]=1.
    return coefficients,normalization,train,val,dict(reports,design_condition=float(singular[0]/singular[-1]),rank=3,
        training_episodes=episodes[~np.isin(episodes,list(heldout))].tolist(),validation_episodes=sorted(heldout))


def acquire(s,c):
    nominal=NominalRobot(s);rng=np.random.default_rng(s['seed']+101)
    rows=[];physics_rows=[];contact_rows=[];examples=[];episode=0;physical=0;true_solves=0;audit_solves=0;matrix_coupling=0.
    block=min(s['calibration_episode_steps'],max(1,s['calibration_decisions']//4))
    while len(rows)<s['calibration_decisions']:
        positions=c['path_calibration_reset_positions'];velocity=c['path_calibration_velocity_scales']
        g=c['speed_train'][episode%len(c['speed_train'])]
        env=PhysicalEnvironment(c,s,700000000+s['seed']*10000+episode,episode,nominal=nominal,purpose='calibration',
            start_x=positions[(episode//len(c['speed_train']))%len(positions)],
            initial_speed=g*velocity[(episode//(len(positions)*len(c['speed_train'])))%len(velocity)])
        try:
            for _ in range(block):
                before=env.current;action=rng.uniform(-1,1,6);requested=env.hold()
                successor,info,label=env.step(action,calibrate=True)
                row=dict(raw=before['raw'],next_raw=successor['raw'],action=action.astype(np.float32),
                    R=info['R'],r_src=info['r_src'],h=info['h'],h_requested=requested,next_h=info['next_h'],
                    done=float(info['terminated'] or info['truncated']),episode=episode)
                for key in BASIS:row[key]=before['basis'][key];row['next_'+key]=successor['basis'][key]
                if 'scan' in before:row.update(scan=before['scan'],next_scan=successor['scan'])
                rows.append(row);physical+=info['physics_steps']
                audit_solves+=info.get('contact_audit',{}).get('audit_forward_solves',0)
                for sample in label['samples']:
                    contacts=sample.pop('contacts',[])
                    for record in contacts:contact_rows.append(dict(record,sample=len(physics_rows)))
                    # Control scans and physical scans have distinct sample axes.
                    if 'scan' in sample:sample['physics_scan']=sample.pop('scan')
                    physics_rows.append(dict(sample,physics_episode=episode))
                matrix=before['basis']['M0'];matrix_coupling=max(matrix_coupling,float(np.linalg.norm(matrix-np.diag(np.diag(matrix)))))
                if len(examples)<24 and label['contacts']:
                    examples.append(dict(episode=episode,time=label['contact_time'],q=label['contact_q'],
                        contacts=label['contacts'],other_constraint_port=label['other_constraint_port']))
                if len(rows)>=s['calibration_decisions'] or row['done']:break
        finally:true_solves+=env.sensor_solves;env.close()
        episode+=1
    if not physics_rows:raise RuntimeError('Calibration collected zero physics labels')
    data={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
    data.update({key:np.asarray([r[key] for r in physics_rows]) for key in physics_rows[0]})
    if contact_rows:data.update({'contact_'+key:np.asarray([r[key] for r in contact_rows]) for key in contact_rows[0]})
    metadata=dict(episodes=episode,nominal_xml_sha256=nominal.xml_sha256,max_off_diagonal_mass_norm=matrix_coupling,
        counters=dict(decisions=len(rows),physics_steps=physical,physics_label_samples=len(physics_rows),
            force_sensor_forward_solves=true_solves,contact_audit_forward_solves=audit_solves,nominal_model_queries=nominal.queries),
        versions=versions())
    return data,examples,metadata


def prepare(s,c,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    report_path=directory/'calibration.json';identity=calibration_id(s,c)
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['calibration_id']!=identity or report.get('measurement_schema')!=measurement_schema(s):
            raise ValueError('Physical calibration contract changed; select a new artifact directory')
        return report
    raw_path=directory/'acquisition.json'
    if raw_path.exists():
        metadata=json.loads(raw_path.read_text())
        if metadata['calibration_id']!=identity or metadata['measurement_schema']!=measurement_schema(s):
            raise ValueError('Saved acquisition has a different calibration contract')
        with np.load(directory/'data.npz') as f:data={k:f[k] for k in f.files}
        examples=json.loads((directory/'contact_examples_raw.json').read_text())
    else:
        data,examples,metadata=acquire(s,c)
        metadata.update(calibration_id=identity,measurement_schema=measurement_schema(s))
        np.savez_compressed(directory/'data.npz',**data)
        write_json(directory/'contact_examples_raw.json',examples);write_json(raw_path,metadata)
    try:
        coefficients,norm,train,val,metrics=fit(data)
        if metrics['validation']['relative_force_rmse']>.05:
            raise ValueError(f"Held-out physical force error exceeds 5%; inspect sensors or extend the template. Metrics: {metrics['validation']}")
    except Exception as error:
        write_json(directory/'calibration_failure.json',dict(calibration_id=identity,error_type=type(error).__name__,
            message=str(error),raw_data_preserved=True))
        raise
    for row in examples:
        for contact in row['contacts']:contact['calibrated_acceleration_response']=(np.asarray(contact['nominal_acceleration_response'])/coefficients[0]).tolist()
    np.savez_compressed(directory/'normalization.npz',**norm,train_indices=np.flatnonzero(train),validation_indices=np.flatnonzero(val))
    report=dict(metadata,schema=s['schema'],control=c,settings=s,coefficients=coefficients.tolist(),
        coefficient_names=['mass_scale','stiffness_scale','damping_scale'],raw_observation_dimension=int(data['raw'].shape[1]),ph_observation_dimension=int(data['raw'].shape[1]+4),
        nominal_mass_scale=s['nominal_mass_scale'],nominal_stiffness_scale=s['nominal_stiffness_scale'],nominal_damping_scale=s['nominal_damping_scale'],metrics=metrics,
        measurement_mode='instrumented exact q, v, acceleration, actuator torque, and constraint-force ports',
        policy_measurements='q, v, constraint-force ports, goal, remaining time, target speed, requested hold; acceleration is calibration-only',
        learned_scope='three positive coefficients around a public XML template; environment force law is measured',
        numerical_value_inverse_used=False,policy_updates=0)
    if 'sensing' in s:
        from environment_belief.config import contract
        report['information_contract']=contract(s)
        report['policy_measurements']='Versioned original-state / environment-belief contract: '+s['sensing']['mode']
    elif s.get('pure',{}).get('terrain_probes'):
        report['policy_measurements']+='; local noisy height/slope probes (same input contract for every method with these options)'
    write_json(directory/'contact_examples.json',examples);write_json(report_path,report)
    (directory/'calibration_failure.json').unlink(missing_ok=True)
    return report


def load(directory):
    directory=Path(directory);report=json.loads((directory/'calibration.json').read_text())
    with np.load(directory/'data.npz') as f:data={k:f[k] for k in f.files}
    with np.load(directory/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    return report,data,norm
