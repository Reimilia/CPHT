"""Training and recovery for both measured-data cases."""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from bench.protocol import write_json,versions
from bench.training import atomic_torch_save
from .agent import Agent
from .calibration import load_prepared,fit_operator_step
from .config import run_key,source_record,label,is_path_task
from .data import Replay,initial_history,next_history
from .environment import Environment
from .models import tensors,row_targets


def train(c,method,prepared_path,directory,device,resume=False,pause_hours=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(c["seed"]);np.random.seed(c["seed"]);random.seed(c["seed"])
    packet,information,model,nodes,node_info=load_prepared(prepared_path,c,device)
    agent=Agent(c,method,information,model,nodes,node_info,device)
    optimizers=[torch.optim.Adam(net.parameters(),lr=c["lr"]) for net in model.members]
    replay=Replay.restore(packet["replay"],c["capacity"])
    validation_replay=Replay.restore(packet["replay"])
    forbidden_calibration=set(packet["validation_indices"].tolist())
    # Serial numbers equal the original calibration indices because that replay
    # has never wrapped. Validation episodes remain excluded from every loss.
    replay.exclude_serials(forbidden_calibration)
    counters=dict(packet["counters"],calibration_decisions=packet["counters"]["decisions"],
        policy_decisions=0,critic_steps=0,actor_steps=0,operator_calls=0,linear_solves=0,
        fit_rejections=0,model_rejections=0,model_member_steps=0,policy_seconds=0.,update_seconds=0.,model_update_seconds=0.,
        observer_updates=c["observer_updates"],operator_member_updates=c["operator_updates"]*c["ensembles"],
        calibration_acquisition_seconds=packet["acquisition_seconds"],calibration_observer_seconds=packet["observer_seconds"],
        calibration_operator_seconds=packet["operator_seconds"] if method in ("model_sac","interval_euler","ph_sac","ph_direct") else 0.)
    episode=0
    collection=torch.Generator(device=device).manual_seed(c["seed"]+401)
    prepared_sha=hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest()
    if resume:
        saved=torch.load(directory/"latest.pt",map_location=device,weights_only=False)
        if saved["config"]!=c or saved["method"]!=method or saved["calibration_sha256"]!=prepared_sha:
            raise ValueError("Resume configuration, method, or common calibration changed")
        agent.load(saved["agent"]);model.load_state_dict(saved["model"]);model.version=saved["model_version"]
        for op,state in zip(optimizers,saved["model_optimizers"]):op.load_state_dict(state)
        replay=Replay.restore(saved["replay"])
        replay.exclude_serials(forbidden_calibration)
        if "validation_replay_rng" in saved:validation_replay.rng.bit_generator.state=saved["validation_replay_rng"]
        counters,episode=saved["counters"],saved["next_episode"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        collection.set_state(saved["collection_rng"].cpu())
        np.random.set_state(saved["numpy_rng"]);random.setstate(saved["python_rng"])
        log=directory/"train.jsonl"
        if log.exists():
            records=[s for s in log.read_text().splitlines() if json.loads(s)["decisions"]<=counters["decisions"]]
            log.write_text("\n".join(records)+("\n" if records else ""))
        episode_file=directory/"episodes.jsonl"
        if episode_file.exists():
            records=[s for s in episode_file.read_text().splitlines() if json.loads(s)["decisions"]<=counters["decisions"]]
            episode_file.write_text("\n".join(records)+("\n" if records else ""))
        for p in directory.glob("policy_*.pt"):
            if int(p.stem.split("_")[-1])>counters["decisions"]:p.unlink()
    metadata=dict(config=c,method=method,label=label(c,method),versions=versions(),sources=source_record(),
        calibration_sha256=prepared_sha,calibration_validation=packet["validation"],backend=packet["backend"],
        observer_kind=packet["observer_kind"],observer_frozen=True,branch="CT" if method.startswith("ct_") else "Op",
        claims=dict(frozen_finite_solver=True,global_filter_bound=False,physical_pH_identification=False,
                    physical_passivity_certificate=False,policy_optimality_certificate=False),
        operator_update=c["model_update"],device=device)
    write_json(directory/"run.json",metadata)
    stop={"requested":False}
    def stopping(*_):stop["requested"]=True
    for sig in (signal.SIGINT,signal.SIGTERM,getattr(signal,"SIGUSR1",signal.SIGTERM)):signal.signal(sig,stopping)
    started=time.perf_counter();previous_wall=counters["policy_seconds"]
    last_save=counters["decisions"]
    def save(path,boundary=False):
        counters["policy_seconds"]=previous_wall+time.perf_counter()-started
        public={k:packet[k] for k in ("config","information","information_mean","information_scale","predictor",
            "reward_scale","nodes","validation","backend","observer_kind")}
        public.update(model=model.state_dict(),model_version=model.version)
        value=dict(config=c,method=method,agent=agent.state(),public=public,counters=counters.copy(),
                   calibration_sha256=prepared_sha,metadata=metadata)
        if boundary:value.update(replay=replay.state(),next_episode=episode,model=model.state_dict(),model_version=model.version,
            model_optimizers=[o.state_dict() for o in optimizers],torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            collection_rng=collection.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            validation_replay_rng=validation_replay.rng.bit_generator.state)
        atomic_torch_save(value,path)
    with (directory/"train.jsonl").open("a",buffering=1) as log,(directory/"episodes.jsonl").open("a",buffering=1) as episode_log:
        while counters["decisions"]<c["total_decisions"]:
            progress=counters["policy_decisions"]/max(1,c["total_decisions"]-c["calibration_decisions"])
            env=Environment(c,1_000_000_000+c["seed"]*10000+episode,episode,progress=progress)
            window=initial_history(env.observation,c["history"])
            observation=env.observation
            previous_action,previous_reward,previous_h=np.zeros(6,np.float32),0.,0.
            counters["observations"]+=1
            sensed=c["experiment"]=="contact" and c["contact_sensor"]!="history"
            counters["environment_sensor_samples"]+=int(sensed)
            counters["odometry_samples"]+=int(is_path_task(c))
            try:
                while counters["decisions"]<c["total_decisions"]:
                    requested=env.hold()
                    action=agent.act(window,env.g,requested,env.base.cur_t,generator=collection)
                    before=env.base.cur_t
                    successor,info=env.step(action)
                    record=dict(obs=observation,next_obs=successor,prev_action=previous_action,action=action,
                        prev_reward=previous_reward,prev_h=previous_h,r_src=info["r_src"],R=info["R"],h=info["h"],
                        next_h=info["next_h"],terminated=info["terminated"],truncated=info["truncated"],
                        g=env.g,time=before,next_time=env.base.cur_t)
                    if is_path_task(c):record["h_requested"]=requested
                    replay.add(record,100_000+episode)
                    for key in ("decisions","policy_decisions","observations"):counters[key]+=1
                    counters["environment_sensor_samples"]+=int(sensed)
                    counters["odometry_samples"]+=int(is_path_task(c))
                    counters["contact_audit_forward_solves"]+=info.get("contact_audit",{}).get("audit_forward_solves",0)
                    counters["physics_steps"]+=info["physics_steps"];counters["reward_samples"]+=info["physics_steps"]
                    counters["simulated_seconds"]+=info["h"]
                    model_stats={}
                    if c["model_update"]=="online" and not method.startswith("ct_") and method!="interval_sac" and counters["policy_decisions"]%c["online_model_every"]==0:
                        if device.startswith("cuda"):torch.cuda.synchronize()
                        model_start=time.perf_counter()
                        old_model=deepcopy(model.state_dict());old_version=model.version
                        old_opts=[deepcopy(op.state_dict()) for op in optimizers]
                        validation_batch=tensors(validation_replay.sample(c["batch"],packet["validation_indices"]),device)
                        with torch.no_grad():
                            validation_info=information.encode_batch(validation_batch)
                            validation_rows,_=row_targets(information.encode_batch(validation_batch,True),validation_batch["g"],node_info,model.node_g,c)
                            def score():
                                vp,vr,vs=model.predict_with_survival(validation_info,validation_batch["action"],validation_batch["g"])
                                labels=torch.cat([validation_batch["r_src"],validation_batch["R"]],1)
                                row_losses=-(validation_rows*vp.clamp_min(1e-12).log()).sum(1)
                                if is_path_task(c):
                                    live=1-torch.maximum(validation_batch["terminated"],validation_batch["truncated"])
                                    loss=(row_losses*live.reshape(-1)).sum()/live.sum().clamp_min(1)
                                    loss=loss+torch.nn.functional.binary_cross_entropy(vs.clamp(1e-6,1-1e-6),live)
                                else:loss=row_losses.mean()
                                loss=loss+((vr-labels)/model.reward_scale).square().mean()
                                return float(loss)
                            before_score=score()
                        model_stats=fit_operator_step(model,information,node_info,model.node_g,replay,None,c,optimizers,device)
                        with torch.no_grad():after_score=score()
                        accepted=np.isfinite(after_score) and after_score<=before_score*(1+c["operator_validation_growth"])+1e-10
                        if not accepted:
                            model.load_state_dict(old_model);model.version=old_version
                            for op,state in zip(optimizers,old_opts):op.load_state_dict(state)
                            counters["model_rejections"]+=1
                        model_stats.update(validation_before=before_score,validation_after=after_score,
                                           model_admitted=bool(accepted),admission_scope="held-out empirical score")
                        if device.startswith("cuda"):torch.cuda.synchronize()
                        counters["model_update_seconds"]+=time.perf_counter()-model_start
                        counters["model_member_steps"]+=len(model.members)
                    if device.startswith("cuda"):torch.cuda.synchronize()
                    start=time.perf_counter()
                    statistics=agent.update(tensors(replay.sample(c["batch"]),device))
                    if device.startswith("cuda"):torch.cuda.synchronize()
                    counters["update_seconds"]+=time.perf_counter()-start
                    for key in ("critic_steps","actor_steps","operator_calls","linear_solves"):counters[key]+=statistics[key]
                    counters["fit_rejections"]+=int(not statistics["critic_admitted"])
                    if counters["decisions"]%c["log_every"]==0:
                        record=dict(counters,diagnostics=statistics,model_fit=model_stats,model_version=model.version,
                            contact_difficulty=env.contact_map["difficulty"] if env.contact_map is not None else None,
                            residual_momentum_reset=True,reset_reason="new actor/model/quadrature block; nodes and encoder fixed")
                        log.write(json.dumps(record,allow_nan=False)+"\n")
                    if counters["decisions"]%c["save_every"]==0:save(directory/f"policy_{counters['decisions']:09d}.pt")
                    window=next_history(window,successor,action,info["r_src"],info["h"])
                    observation=successor
                    previous_action,previous_reward,previous_h=action,info["r_src"],info["h"]
                    if pause_hours and time.perf_counter()-started>=pause_hours*3600:stop["requested"]=True
                    if info["terminated"] or info["truncated"]:break
            finally:
                if env.path_task is not None:
                    episode_log.write(json.dumps(dict(episode=episode,decisions=counters["decisions"],
                        complete=env.path_task.reason is not None,path_metrics=env.path_task.summary(False)),allow_nan=False)+"\n")
                env.close()
            episode+=1
            if stop["requested"] or counters["decisions"]-last_save>=c["save_every"]:
                save(directory/"latest.pt",True);last_save=counters["decisions"]
            if stop["requested"]:break
    complete=counters["decisions"]>=c["total_decisions"]
    if complete:save(directory/"final.pt")
    status=dict(status="complete" if complete else "paused",counters=counters,
                primary_checkpoint="final.pt" if complete else None,episode_boundary_resume="latest.pt")
    write_json(directory/"status.json",status)
    return status
