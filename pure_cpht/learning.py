"""Positive mechanics regression and measured contact closure calibration."""
import numpy as np
import torch
from scipy.optimize import lsq_linear
from .mechanics import Robot,rotate,public_geometry
from .contact import Contact


def prior_parameters(geometry,coefficients):
    a,k,d=coefficients
    return np.asarray(sum([geometry[name] for name in ('masses','inertias','armature','stiffness','damping')],[]))*np.repeat([a,k,d],[20,6,6])


def split_masks(data,validation_episodes):
    val=np.isin(data['physics_episode'],validation_episodes)
    if val.all() or not val.any():raise ValueError('Pure dynamics learning needs disjoint episode folds')
    return ~val,val


def fit_mechanics(geometry,data,prior,validation_episodes,options):
    robot=Robot(geometry,dtype=torch.float64);t=robot.t
    x=robot.regressor(t(data['q']),t(data['v']),t(data['acceleration'])).numpy();y=data['y']
    train,val=split_masks(data,validation_episodes)
    row_scale=np.sqrt(np.mean(y[train]**2,axis=0)).clip(1.)
    design=(x[train]*prior[None,None,:]/row_scale[None,:,None]).reshape(-1,32)
    rhs=(y[train]/row_scale).reshape(-1)
    ridge=options['mass_ridge'];rank=int(np.linalg.matrix_rank(design))
    fit=lsq_linear(np.vstack((design,np.sqrt(ridge)*np.eye(32))),np.r_[rhs,np.sqrt(ridge)*np.ones(32)],
        bounds=(.05,20.),tol=1e-11)
    if not fit.success:raise RuntimeError('Positive mechanical calibration failed: '+fit.message)
    parameters=prior*fit.x
    metrics={}
    for name,mask in (('train',train),('validation',val)):
        residual=x[mask]@parameters-y[mask];initial=x[mask]@prior-y[mask]
        metrics[name]=dict(force_rmse=float(np.sqrt(np.mean(residual**2))),
            relative_force_rmse=float(np.sqrt(np.mean(residual**2))/max(1e-8,np.sqrt(np.mean(y[mask]**2)))),
            prior_force_rmse=float(np.sqrt(np.mean(initial**2))),power_rmse=float(np.sqrt(np.mean((residual*data['v'][mask]).sum(1)**2))),samples=int(mask.sum()))
    if metrics['validation']['relative_force_rmse']>.05:raise ValueError('Held-out mechanical force error exceeds 5%; preserve data and revise the physical template')
    metrics.update(design_rank=rank,parameters=32,regularized_identification=True,
        parameter_identifiability='Only identifiable combinations are data-determined; ridge selects remaining coefficients near the positive prior')
    return parameters,metrics


def fit_contact(robot,data,validation_episodes,options):
    if 'contact_sample' not in data:raise ValueError('Contact labels are absent; prepare with Pure CPHT options')
    indices=data['contact_sample'];body=data['contact_body'].astype(int);t=robot.t
    q=t(data['q'][indices]);v=t(data['v'][indices]);acc=t(data['acceleration'][indices]);local=t(data['contact_local'])
    kin=robot.kinematics(q,v);rows=torch.arange(len(q),device=q.device)
    offset=rotate(kin['angles'][rows,body],local)
    bias=kin['origin_bias'][rows,body]-kin['omega'][rows,body,None]**2*offset
    J=t(data['contact_jacobian']);normal=t(data['contact_normal']);force=t(data['contact_force'])
    velocity=torch.einsum('ndi,ni->nd',J,v);an=(normal*(torch.einsum('ndi,ni->nd',J,acc)+bias)).sum(1)
    vn=(normal*velocity).sum(1);fn=(normal*force).sum(1)
    tangent=torch.stack((normal[:,1],-normal[:,0]),1);vt=(tangent*velocity).sum(1);ft=(tangent*force).sum(1)
    normal_J=torch.einsum('nd,ndi->ni',normal,J)
    mass=robot.quantities(q,v)['M'];response=torch.linalg.solve(mass,normal_J[:,:,None]).squeeze(-1)
    inverse_contact_mass=(normal_J*response).sum(1)
    design=torch.stack((-t(data['contact_gap']),-vn,-inverse_contact_mass*fn),1).numpy();target=an.numpy()
    train,validation=split_masks(data,validation_episodes);good=fn.numpy()>1.
    mask=train[indices]&good;val=validation[indices]&good
    prior=np.array([2500.,100.,.08]);theta=prior.copy();calibrated=False
    if mask.sum()>=6:
        scale=max(1.,float(np.sqrt(np.mean(target[mask]**2))));A=design[mask]*prior/scale;y=target[mask]/scale
        ridge=options['contact_ridge']
        fitted=lsq_linear(np.vstack((A,np.sqrt(ridge)*np.eye(3))),np.r_[y,np.sqrt(ridge)*np.ones(3)],
            bounds=([.05,.05,.01],[20.,20.,100.]),tol=1e-10)
        if fitted.success:theta=prior*fitted.x;calibrated=True
    slip=mask&(abs(vt.numpy())>.1)
    mu=float(np.clip(np.quantile(abs(ft.numpy()[slip])/fn.numpy()[slip],.8),.05,1.5)) if slip.sum()>=6 else .6
    metric=dict(calibrated=calibrated,training_contacts=int(mask.sum()),validation_contacts=int(val.sum()),
        normal_reference_validation_rmse=float(np.sqrt(np.mean((design[val]@theta-target[val])**2))) if val.any() else None,
        friction_from_sliding_samples=int(slip.sum()),parameter_names=['normal_stiffness','normal_damping','impedance_ratio','friction'],
        approximation='Finite-iteration unilateral Coulomb-cone contact QP with measured local terrain; separate from the MuJoCo solver')
    return np.r_[theta,mu],metric


def fit_contact_ports(robot,data,validation_episodes,options):
    """Fit the contact closure from original-state ports and causal scans.

    No contact point, surface normal, gap, friction label, or point-force label
    from the simulator enters this fit. The public robot geometry supplies the
    candidate points and their Jacobians.
    """
    from scipy.optimize import least_squares
    train, validation=split_masks(data,validation_episodes)
    ix=np.flatnonzero(train)
    if len(ix)>128:ix=ix[np.linspace(0,len(ix)-1,128,dtype=int)]
    t=robot.t;q=t(data['q'][ix]);v=t(data['v'][ix]);u=t(data['u'][ix]);scan=t(data['physics_scan'][ix])
    target=np.asarray(data['constraint_port'][ix]);scale=np.sqrt(np.mean(target**2,0)).clip(10.)
    prior=np.array([2500.,100.,.08,.6]);ridge=options['contact_ridge']
    with torch.no_grad():
        quantities=robot.quantities(q,v)
        def residual(ratio):
            closure=Contact(robot,prior*ratio,options)
            prediction=closure.solve(q,v,u,scan,quantities)['port'].numpy()
            return np.r_[((prediction-target)/scale).ravel()/max(1.,len(ix)**.5),np.sqrt(ridge)*(ratio-1.)]
        initial=residual(np.ones(4))
        fitted=least_squares(residual,np.ones(4),bounds=([.05,.05,.01,.05],[20.,20.,100.,4.]),
            max_nfev=30,diff_step=1e-3,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    admitted=bool(np.isfinite(fitted.x).all() and np.dot(fitted.fun,fitted.fun)<=np.dot(initial,initial))
    parameters=prior*(fitted.x if admitted else np.ones(4))
    metrics=dict(calibrated=admitted,supervision='aggregate_ports',training_samples=len(ix),
        validation_samples=int(validation.sum()),optimizer_converged=bool(fitted.success),optimizer_evaluations=int(fitted.nfev),
        point_contact_labels_used=False,friction_labels_used=False,
        parameter_names=['normal_stiffness','normal_damping','impedance_ratio','friction'],
        approximation='Positive finite-iteration contact closure fitted to measured aggregate ports and causal terrain estimates')
    return parameters,metrics


def fit_all(nominal,calibration,data,options,previous=None):
    geometry=public_geometry(nominal) if previous is None else previous['geometry']
    prior=prior_parameters(geometry,calibration['coefficients']) if previous is None else np.asarray(previous['parameters'])
    validation=calibration['metrics']['validation_episodes']
    parameters,metrics=fit_mechanics(geometry,data,prior,validation,options)
    robot=Robot(geometry,parameters,dtype=torch.float64)
    if options.get('contact_supervision')=='aggregate_ports':
        contact,contact_metrics=fit_contact_ports(robot,data,validation,options)
    else:contact,contact_metrics=fit_contact(robot,data,validation,options)
    model=dict(geometry=geometry,parameters=parameters.tolist(),contact=contact.tolist())
    # Evaluate actual constraint-port prediction, independent of the mechanics fit.
    _,mask=split_masks(data,validation);ix=np.flatnonzero(mask)[:128]
    with torch.no_grad():
        closure=Contact(robot,contact,options);pred=closure.solve(robot.t(data['q'][ix]),robot.t(data['v'][ix]),robot.t(data['u'][ix]),robot.t(data['physics_scan'][ix]))
        residual=pred['port'].numpy()-data['constraint_port'][ix]
        if previous is not None:
            old=Contact(robot,previous['contact'],options).solve(robot.t(data['q'][ix]),robot.t(data['v'][ix]),robot.t(data['u'][ix]),robot.t(data['physics_scan'][ix]))
            old_residual=old['port'].numpy()-data['constraint_port'][ix]
            admitted=np.mean(residual**2)<=1.1**2*np.mean(old_residual**2)
            contact_metrics['candidate_admitted']=bool(admitted)
            contact_metrics['candidate_port_rmse']=float(np.sqrt(np.mean(residual**2)))
            if not admitted:
                model['contact']=previous['contact'];pred=old;residual=old_residual
    contact_metrics.update(port_validation_rmse=float(np.sqrt(np.mean(residual**2))),
        port_validation_relative_rmse=float(np.sqrt(np.mean(residual**2))/max(1.,np.sqrt(np.mean(data['constraint_port'][ix]**2)))),
        qp_projected_residual=float(pred['projected_residual'].max()),validation_samples=len(ix))
    return model,dict(mechanics=metrics,contact=contact_metrics)


def prepare_artifact(directory,settings=None):
    import json
    from pathlib import Path
    from bench.protocol import digest,write_json
    from physical_fj.prepare import load
    from physical_fj.mechanics import NominalRobot
    directory=Path(directory);calibration,data,norm=load(directory)
    from .config import options as validate_options
    s=dict(calibration['settings'] if settings is None else settings)
    if 'pure' not in s:raise ValueError('Calibration settings need a "pure" block (see experiments/*.json)')
    s['pure']=validate_options(s['pure']);torch.set_num_threads(1)
    signature=digest(dict(schema='pure_cpht_mechanics_v2',calibration=calibration['calibration_id'],options=s['pure']))[:12]
    path=directory/('pure_model_'+signature+'.json')
    if path.exists():return json.loads(path.read_text())
    nominal=NominalRobot(s);model,metrics=fit_all(nominal,calibration,data,s['pure'])
    first=np.r_[True,data['episode'][1:]!=data['episode'][:-1]]
    neutral=np.median(data['raw'][first,3:9],axis=0).tolist()
    result=dict(schema='pure_cpht_mechanics_v2',signature=signature,calibration_id=calibration['calibration_id'],
        model=model,metrics=metrics,neutral=neutral,actor_updates=0,critic_updates=0)
    write_json(path,result)
    return result
