"""Common measured-data acquisition, history calibration, and operator fitting."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import Replay
from .environment import Environment
from .models import InformationMap, ObservationPredictor, OperatorModel, make_nodes, row_targets, tensors
from .config import observation_dim,is_path_task
from bench.protocol import write_json, versions
from bench.training import atomic_torch_save


def acquire(c,verbose=False):
    obs_dim=observation_dim(c)
    replay=Replay(c["calibration_decisions"],obs_dim,c["history"],c["seed"]+151,include_requested_hold=is_path_task(c))
    rng=np.random.default_rng(c["seed"]+157)
    episode=0
    counters=dict(decisions=0,physics_steps=0,simulated_seconds=0.,observations=0,reward_samples=0)
    counters["environment_sensor_samples"]=0
    counters["contact_audit_forward_solves"]=0
    counters["odometry_samples"]=0
    exposure=[];path_outcomes=[];backend={}
    path=is_path_task(c)
    targets=list(dict.fromkeys(map(float,c["speed_train"])))
    goals={g:0 for g in targets};attempts={g:0 for g in targets}
    quota=c.get("path_calibration_goal_episodes",4)
    attempt_limit=c.get("path_calibration_goal_attempts",32)
    probe_steps=c.get("path_calibration_probe_steps",4)
    counters.update(calibration_goal_probe_decisions=0,calibration_goal_probe_episodes=0)
    while replay.count<c["calibration_decisions"]:
        pending=[g for g in targets if goals[g]<quota and attempts[g]<attempt_limit] if path else []
        probe=bool(pending);extra={};task_episode=episode
        if probe:
            g=pending[0];attempts[g]+=1
            task_episode=c["speed_train"].index(g)
            margin=min(.01*c["path_length"],.25*g*c["physics_dt"])
            extra=dict(start_x=c["path_start_x"]+c["path_length"]-margin,initial_speed=g)
            counters["calibration_goal_probe_episodes"]+=1
        elif path and min(goals.values())<2:
            # Stop early when bounded physical probes cannot supply two
            # distinct goal episodes per task. Preparation saves this evidence.
            break
        levels=c["contact_calibration_levels"]
        env=Environment(c,700_000_000+c["seed"]*10000+episode,task_episode,
                        difficulty=levels[(episode//len(c["speed_train"]))%len(levels)] if c["case"]=="J" else 1.,purpose="calibration",**extra)
        previous_action,previous_reward,previous_h=np.zeros(6,np.float32),0.,0.
        observation=env.observation
        counters["observations"]+=1
        backend=env.backend
        sensed=c["experiment"]=="contact" and c["contact_sensor"]!="history"
        counters["environment_sensor_samples"]+=int(sensed)
        counters["odometry_samples"]+=int(is_path_task(c))
        phase="goal_probe" if probe else "ordinary"
        if env.contact_map is not None:exposure.append(dict(episode=episode,g=env.g,collection_phase=phase,**env.audit_truth()))
        episode_steps=0
        try:
            while replay.count<c["calibration_decisions"]:
                action=rng.uniform(-1,1,6).astype(np.float32)
                start=env.base.cur_t
                requested=env.hold()
                successor,info=env.step(action)
                record=dict(obs=observation,next_obs=successor,prev_action=previous_action,
                    action=action,prev_reward=previous_reward,prev_h=previous_h,
                    r_src=info["r_src"],R=info["R"],h=info["h"],next_h=info["next_h"],
                    terminated=info["terminated"],truncated=info["truncated"],g=env.g,
                    time=start,next_time=env.base.cur_t)
                if is_path_task(c):record["h_requested"]=requested
                replay.add(record,episode)
                counters["decisions"]+=1;counters["observations"]+=1
                counters["environment_sensor_samples"]+=int(sensed)
                counters["odometry_samples"]+=int(is_path_task(c))
                counters["contact_audit_forward_solves"]+=info.get("contact_audit",{}).get("audit_forward_solves",0)
                counters["physics_steps"]+=info["physics_steps"]
                counters["reward_samples"]+=info["physics_steps"]
                counters["simulated_seconds"]+=info["h"]
                episode_steps+=1
                if probe:counters["calibration_goal_probe_decisions"]+=1
                observation=successor
                previous_action,previous_reward,previous_h=action,info["r_src"],info["h"]
                if info["terminated"] or info["truncated"]:break
                if probe and episode_steps>=probe_steps:break
                if verbose and counters["decisions"]%1000==0:
                    print(f"Calibration {c['case']}: {counters['decisions']}/{c['calibration_decisions']} decisions; goal episodes {goals}",flush=True)
        finally:
            if env.path_task is not None:
                outcome=dict(episode=episode,collection_phase=phase,collection_decisions=episode_steps,
                    complete=env.path_task.reason is not None,**env.path_task.summary(False))
                path_outcomes.append(outcome)
                if outcome["success"]:goals[outcome["target_speed"]]+=1
            env.close()
        episode+=1
    if exposure:backend=dict(backend,calibration_contact_exposure=exposure)
    if path_outcomes:backend=dict(backend,path_calibration_outcomes=path_outcomes)
    backend=dict(backend,calibration_protocol=c.get("calibration_protocol","goal_balanced_v2" if path else "episode_v1"),
        acquisition_complete=replay.count==c["calibration_decisions"])
    if path:backend["path_goal_probes"]=dict(target_per_task=quota,attempt_limit=attempt_limit,step_limit=probe_steps,
        goal_episodes={str(k):v for k,v in goals.items()},attempts={str(k):v for k,v in attempts.items()})
    return replay,counters,backend


def partition_acquisition(c,replay,backend):
    """Return episode-disjoint indices and a diagnostic report before fitting."""
    path=is_path_task(c);outcomes=backend.get("path_calibration_outcomes",[])
    strata={r["episode"]:(float(r["target_speed"]),r["terminal_reason"]) for r in outcomes} if path else None
    error=None
    try:train,val=replay.episode_split(strata=strata)
    except ValueError as exc:
        train=val=np.empty(0,dtype=int);error=str(exc)
    goals={r["episode"] for r in outcomes if r["success"]}
    coverage={name:len(goals & set(replay.episodes[ix].tolist())) for name,ix in (("train",train),("validation",val))}
    by_task={}
    for g in c["speed_train"]:
        task_goals={r["episode"] for r in outcomes if r["success"] and np.isclose(r["target_speed"],g)}
        by_task[str(float(g))]={name:len(task_goals & set(replay.episodes[ix].tolist())) for name,ix in (("train",train),("validation",val))}
    train_tasks=np.unique(replay.arrays["g"][train]) if len(train) else np.empty(0)
    missing=[float(g) for g in c["speed_train"] if not np.isclose(train_tasks,g).any()]
    complete=backend.get("acquisition_complete",True) and replay.count==c["calibration_decisions"]
    admitted=(error is None and complete and not missing and
        (not path or all(min(counts.values())>=1 for counts in by_task.values())))
    report=dict(protocol=backend.get("calibration_protocol"),admitted=admitted,split_error=error,
        decisions=replay.count,expected_decisions=c["calibration_decisions"],episodes=len(np.unique(replay.episodes[replay.indices()])),
        training_records=len(train),validation_records=len(val),goal_episodes_collected=len(goals),
        path_goal_episode_coverage=coverage,path_goal_coverage_by_task=by_task,missing_training_tasks=missing,
        acquisition_complete=complete,goal_probes=backend.get("path_goal_probes"),
        split_policy="task/outcome strata; whole episodes" if path else "every fifth episode")
    return train,val,report


def fit_operator_step(model,information,node_information,node_g,replay,indices,c,optimizers,device):
    statistics=[]
    for k,optimizer in enumerate(optimizers):
        batch=tensors(replay.sample(c["batch"],indices),device)
        with torch.no_grad():
            current=information.encode_batch(batch)
            next_i=information.encode_batch(batch,True)
            rows,_=row_targets(next_i,batch["g"],node_information,node_g,c)
            rewards=torch.cat([batch["r_src"],batch["R"]],1)
            gamma=torch.exp(np.log(c["gamma_reference"])*batch["h"]/c["reference_time"])
            raw_gamma=torch.exp(np.log(c["gamma_reference"])*batch.get("h_requested",batch["h"])/c["reference_time"])
            done=batch["terminated"]
            if c["boundary"]=="finite_horizon":done=torch.maximum(done,batch["truncated"])
            gamma=gamma*(1-done)
            probes=(torch.rand(len(node_g),c["probe_count"],device=device)>.5).float()*2-1
        optimizer.zero_grad(set_to_none=True)
        extra=dict(survival_target=1-done,raw_gamma=raw_gamma) if is_path_task(c) else {}
        loss,stats=model.loss(current,batch["action"],batch["g"],rows,rewards,gamma,probes,k,**extra)
        loss.backward();optimizer.step()
        statistics.append(stats)
    model.version+=1
    return {key:float(np.mean([s[key] for s in statistics])) for key in statistics[0]}


def prepare(c,directory,device):
    from .config import calibration_key
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"prepared.pt").exists():raise FileExistsError("Calibration artifact exists; reuse it or select a new root")
    torch.manual_seed(c["seed"])
    started=time.perf_counter()
    acquired=directory/"acquired.pt";reused=acquired.exists()
    if reused:
        raw=torch.load(acquired,map_location="cpu",weights_only=False)
        if raw.get("acquisition_schema")!=2 or calibration_key(raw["config"])!=calibration_key(c):
            raise ValueError("Saved acquisition and current calibration contract differ; use the matching preparation directory")
        replay=Replay.restore(raw["replay"]);counters=raw["counters"];backend=raw["backend"]
        acquisition_seconds=raw["acquisition_seconds"];fit_started=started
        print(f"Reusing measured calibration acquisition: {acquired}",flush=True)
    else:
        replay,counters,backend=acquire(c,verbose=True)
        acquisition_seconds=time.perf_counter()-started;fit_started=time.perf_counter()
        atomic_torch_save(dict(acquisition_schema=2,config=c,replay=replay.state(),counters=counters,
            backend=backend,acquisition_seconds=acquisition_seconds),acquired)
    train,val,admission=partition_acquisition(c,replay,backend)
    backend["path_goal_episode_coverage"]=admission["path_goal_episode_coverage"]
    backend["path_goal_coverage_by_task"]=admission["path_goal_coverage_by_task"]
    diagnostic=directory/"calibration_acquisition.json"
    write_json(diagnostic,dict(config=c,counters=counters,admission=admission,backend=backend,
        acquisition_seconds=acquisition_seconds,acquisition_reused=reused,neural_fitting_started=False))
    if not admission["admitted"]:
        raise RuntimeError(f"Calibration coverage failed before fitting: collected {admission['goal_episodes_collected']} goal episodes; "
            f"train/validation coverage by task={admission['path_goal_coverage_by_task']}; "
            f"split_error={admission['split_error']}; complete_budget={admission['acquisition_complete']}. "
            f"Measured data were saved to {acquired}. See {diagnostic} for probe attempts and terminal causes.")
    frames=np.concatenate([replay.arrays[k][train] for k in ("obs","prev_action","prev_reward","prev_h")],1)
    information=InformationMap(c,frames.mean(0),frames.std(0)).to(device)
    predictor=ObservationPredictor(information).to(device)
    optimizer=torch.optim.Adam(predictor.parameters(),lr=c["lr"])
    observer_start=time.perf_counter()
    last={}
    for _ in range(c["observer_updates"]):
        batch=tensors(replay.sample(c["batch"],train),device)
        optimizer.zero_grad(set_to_none=True)
        loss,last=predictor.loss(batch);loss.backward();optimizer.step()
    information.eval().requires_grad_(False)
    if device.startswith("cuda"):torch.cuda.synchronize()
    observer_seconds=time.perf_counter()-observer_start
    nodes=make_nodes(replay,train,c)
    with torch.no_grad():
        node_information=information.encode_batch(tensors(nodes,device))
    reward_scale=np.std(np.concatenate([replay.arrays[k][train] for k in ("r_src","R")],1),0)
    model=OperatorModel(information.feature_dim,nodes,c,reward_scale).to(device)
    optimizers=[torch.optim.Adam(member.parameters(),lr=c["lr"]) for member in model.members]
    model_stats={}
    operator_start=time.perf_counter()
    for _ in range(c["operator_updates"]):
        model_stats=fit_operator_step(model,information,node_information,model.node_g,replay,train,c,optimizers,device)
    validation=[]
    if device.startswith("cuda"):torch.cuda.synchronize()
    operator_seconds=time.perf_counter()-operator_start
    with torch.no_grad():
        for _ in range(8):
            batch=tensors(replay.sample(c["batch"],val),device)
            _,stats=predictor.loss(batch)
            row,nearest=row_targets(information.encode_batch(batch,True),batch["g"],node_information,model.node_g,c)
            p,r,s=model.predict_with_survival(information.encode_batch(batch),batch["action"],batch["g"])
            live=1-torch.maximum(batch["terminated"],batch["truncated"]) if is_path_task(c) else torch.ones_like(batch["terminated"])
            row_error=(abs(p-row).sum(1)*live.reshape(-1)).sum()/live.sum().clamp_min(1)
            stats.update(row_l1=float(row_error),reward_R_mse=float((r[:,1:]-batch["R"]).square().mean()),
                         next_anchor_distance=float(nearest.mean()))
            if is_path_task(c):stats["survival_bce"]=float(torch.nn.functional.binary_cross_entropy(s.clamp(1e-6,1-1e-6),live))
            validation.append(stats)
    validation={k:float(np.mean([v[k] for v in validation])) for k in validation[0]}
    if device.startswith("cuda"):torch.cuda.synchronize()
    packet=dict(config=c,information=information.state_dict(),information_mean=frames.mean(0),
        information_scale=frames.std(0),predictor=predictor.net.state_dict(),model=model.state_dict(),
        model_version=model.version,reward_scale=reward_scale,nodes=nodes,replay=replay.state(),
        train_indices=train,validation_indices=val,counters=counters,backend=backend,
        preparation_seconds=acquisition_seconds+time.perf_counter()-fit_started,validation=validation,
        this_attempt_seconds=time.perf_counter()-started,acquisition_reused=reused,acquisition_admission=admission,
        acquisition_seconds=acquisition_seconds,observer_seconds=observer_seconds,operator_seconds=operator_seconds,
        observer_kind=information.kind,scope="Op predictive representation; empirical validation only")
    atomic_torch_save(packet,directory/"prepared.pt")
    write_json(directory/"calibration.json",dict(config=c,counters=counters,backend=backend,versions=versions(),
        preparation_seconds=packet["preparation_seconds"],validation=validation,
        this_attempt_seconds=packet["this_attempt_seconds"],acquisition_reused=reused,acquisition_admission=admission,
        acquisition_seconds=acquisition_seconds,observer_seconds=observer_seconds,operator_seconds=operator_seconds,
        observer_updates=c["observer_updates"],operator_member_updates=c["operator_updates"]*c["ensembles"],
        stages=dict(observer=last,operator=model_stats),posterior_closure_certificate=None,
        physical_parameter_certificate=None,source="measured sensor/reward records only"))
    return packet


def load_prepared(path,c,device):
    from .config import calibration_key
    packet=path if isinstance(path,dict) else torch.load(path,map_location=device,weights_only=False)
    if calibration_key(packet["config"])!=calibration_key(c):
        raise ValueError("Calibration and experiment information contracts differ")
    information=InformationMap(c,packet["information_mean"],packet["information_scale"]).to(device)
    information.load_state_dict(packet["information"]);information.eval().requires_grad_(False)
    model=OperatorModel(information.feature_dim,packet["nodes"],c,packet["reward_scale"]).to(device)
    model.load_state_dict(packet["model"]);model.version=packet["model_version"]
    nodes=tensors(packet["nodes"],device)
    with torch.no_grad():node_information=information.encode_batch(nodes)
    return packet,information,model,nodes,node_information
