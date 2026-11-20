"""Source-equation baselines and frozen twin-critic interval solvers."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from bench.learner import Actor,mlp
from .finite import entropy,solve,backup,ResidualRecoveryError
from .config import is_path_task


class Agent:
    def __init__(self,c,method,information,model,nodes,node_information,device):
        self.c,self.method,self.device=c,method,torch.device(device)
        self.information,self.model=information,model
        self.nodes,self.node_information=nodes,node_information
        self.actor=Actor(information.feature_dim,6,c["hidden"]).to(device)
        self.critics=nn.ModuleList([mlp(information.feature_dim+6,1,c["hidden"]) for _ in range(2)]).to(device)
        self.targets=deepcopy(self.critics).requires_grad_(False)
        self.target_actor=deepcopy(self.actor).requires_grad_(False)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=c["lr"])
        self.critic_opt=torch.optim.Adam(self.critics.parameters(),lr=c["lr"])
        self.updates=0
        self.last_critic_steps=0

    def q(self,info,action,target=False):
        x=torch.cat([info,action],1)
        return torch.stack([net(x) for net in (self.targets if target else self.critics)])

    def alpha(self,h):
        u=h/self.c["reference_time"]
        if self.c["entropy_convention"]=="start":return self.c["alpha_reference"]*u
        beta=-np.log(self.c["gamma_reference"])
        return self.c["alpha_reference"]*(-torch.expm1(-beta*u))/beta

    def act(self,window,g,h,time,deterministic=False,generator=None):
        with torch.no_grad():
            args=[torch.as_tensor(v,dtype=torch.float32,device=self.device) for v in
                  (window[None],[[g]],[[h]],[[time]])]
            info=self.information(*args)
            is_td3=self.method=="ct_td3"
            action=self.actor.sample(info,deterministic or is_td3,generator)[0][0]
            if is_td3 and not deterministic:
                action=(action+self.c["td3_exploration"]*torch.randn(action.shape,device=self.device,generator=generator)).clamp(-1,1)
            return action.cpu().numpy()

    def finite_block(self):
        c=self.c
        n,k=len(self.node_information),c["action_samples"]
        info=self.node_information.repeat_interleave(k,0)
        actions,logp=self.actor.sample(info)
        g=self.nodes["g"].repeat_interleave(k,0)
        p,rewards,survival=self.model.predict_with_survival(info,actions,g)
        r=rewards[:,1].reshape(n,k)
        h=self.nodes.get("h_requested",self.nodes["h"]).repeat_interleave(k,0)
        gamma=torch.exp(np.log(c["gamma_reference"])*h/c["reference_time"])
        done=self.nodes["terminated"]
        if c["boundary"]=="finite_horizon":done=torch.maximum(done,self.nodes["truncated"])
        gamma=gamma*(survival if is_path_task(c) else 1-done.repeat_interleave(k,0))
        if is_path_task(c):
            at_deadline=(self.nodes["time"]+self.nodes.get("h_requested",self.nodes["h"])).repeat_interleave(k,0)>=c["horizon"]-1e-8
            gamma=gamma*(~at_deadline)
        initial=self.q(info,actions,target=True).reshape(2,n,k)
        ent=(self.alpha(h)*logp).reshape(n,k)
        return dict(info=info,actions=actions,initial=initial,p=p.reshape(n,k,n),r=r,
                    gamma=gamma.reshape(n,k),entropy=ent,
                    signature=dict(model_version=self.model.version,actor_version=self.updates,
                                   encoder_frozen=True,alpha_reference=c["alpha_reference"]))

    def update(self,batch):
        c=self.c
        self.updates+=1
        with torch.no_grad():
            info=self.information.encode_batch(batch)
            next_info=self.information.encode_batch(batch,True)
            hbar=batch["h"]/c["reference_time"]
            gamma=torch.exp(np.log(c["gamma_reference"])*hbar)
            done=batch["terminated"]
            if c["boundary"]=="finite_horizon":done=torch.maximum(done,batch["truncated"])
            continuation=gamma*(1-done)
        method=self.method
        stats=dict(operator_calls=0,linear_solves=0,critic_steps=0,actor_steps=0,
                   numerical_gate_pass=True,phase_gate_pass=True,critic_fit_pass=True,operator_resets=1)
        finite=method in ("interval_euler","ph_sac","ph_direct")
        block=None
        with torch.no_grad():
            if finite:
                block=self.finite_block()
                arrays=[block[key].detach().cpu().double().numpy() for key in
                        ("initial","r","gamma","p","entropy")]
                arrays[3]=arrays[3]/arrays[3].sum(-1,keepdims=True)
                try:target,rho,diagnostic=solve(*arrays,c,method)
                except ResidualRecoveryError as exc:
                    stats.update(exc.diagnostics)
                    stats.update(numerical_gate_pass=False,phase_gate_pass=False,critic_admitted=False,
                        actor_admitted=False,critic_fit_pass=False,complete_numerical_gate_pass=False,
                        solver_failure=str(exc),policy_gap_certificate=None)
                    return stats
                target=torch.as_tensor(target.reshape(2,-1,1),dtype=torch.float32,device=self.device)
                train_info,train_actions=block["info"],block["actions"]
                stats.update(diagnostic)
            elif method=="ct_td3":
                current=self.q(info,batch["action"],True).min(0).values
                action=self.target_actor.sample(next_info,True)[0]
                noise=(torch.randn_like(action)*c["td3_target_noise"]).clamp(-c["td3_noise_clip"],c["td3_noise_clip"])
                following=self.q(next_info,(action+noise).clamp(-1,1),True).min(0).values
                fast=current+(gamma*following-current)/(hbar+1e-8)
                stats["duration_floor_defect"]=float(abs((gamma*following-current)/(hbar+1e-8)-(gamma*following-current)/hbar).max())
                target=batch["r_src"]+(1-done)*fast
            elif method in ("ct_sac","ct_sac_i"):
                a,lp=self.actor.sample(info)
                ap,lpp=self.actor.sample(next_info)
                current=self.q(info,a,True).min(0).values-c["alpha_reference"]*lp
                following=self.q(next_info,ap,True).min(0).values-c["alpha_reference"]*lpp
                reward=batch["r_src"] if method=="ct_sac" else batch["R"]/hbar
                target=reward+(1-done)*(current+(gamma*following-current)/(hbar+1e-8))
                stats["duration_floor_defect"]=float(abs((gamma*following-current)/(hbar+1e-8)-(gamma*following-current)/hbar).max())
            elif method=="interval_sac":
                ap,lpp=self.actor.sample(next_info)
                target=batch["R"]+continuation*(self.q(next_info,ap,True).min(0).values-self.alpha(batch["next_h"])*lpp)
            elif method=="model_sac":
                n,k=len(self.node_information),c["action_samples"]
                ni=self.node_information.repeat_interleave(k,0)
                ap,lpp=self.actor.sample(ni)
                soft=(self.q(ni,ap,True).min(0).values-self.alpha(self.nodes.get("h_requested",self.nodes["h"]).repeat_interleave(k,0))*lpp).reshape(n,k).mean(1)
                p,r,survival=self.model.predict_with_survival(info,batch["action"],batch["g"])
                planned_gamma=torch.exp(np.log(c["gamma_reference"])*batch.get("h_requested",batch["h"])/c["reference_time"])
                discount=planned_gamma*survival if is_path_task(c) else continuation
                if is_path_task(c):discount=discount*((batch["time"]+batch.get("h_requested",batch["h"]))<c["horizon"]-1e-8)
                target=r[:,1:]+discount*(p@soft[:,None])
            else:raise ValueError(method)
            if is_path_task(c) and method.startswith("ct_"):
                # The finite-course terminal event is a value boundary. Its
                # measured interval return is not divided by a shortened hold.
                stats["terminal_boundary_correction"]=float((abs(target-batch["R"])*done).max())
                target=torch.where(done>0,batch["R"],target)
                stats["terminal_target_error"]=float((abs(target-batch["R"])*done).max())
        if not finite:
            train_info,train_actions=info,batch["action"]
            target=target.unsqueeze(0).expand(2,-1,-1)
        old_critics=deepcopy(self.critics.state_dict()) if finite else None
        old_optimizer=deepcopy(self.critic_opt.state_dict()) if finite else None
        steps=c["fit_steps"] if finite else 1
        for _ in range(steps):
            prediction=self.q(train_info,train_actions)
            loss=(prediction-target).square().mean((1,2)).sum()
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Non-finite critic loss")
            self.critic_opt.zero_grad(set_to_none=True);loss.backward();self.critic_opt.step()
            stats["critic_steps"]+=1
        with torch.no_grad():
            fitted=self.q(train_info,train_actions)
            error=float(abs(fitted-target).max())
            stats.update(critic_loss=float(loss),critic_fit_inf=error,
                         fit_tolerance=c["fit_relative_tolerance"]*(1+float(abs(target).max())))
            if finite:
                stats["critic_fit_pass"]=error<=stats["fit_tolerance"]
                qfit=fitted.reshape(arrays[0].shape).cpu().double().numpy()
                fit_residual=qfit-backup(qfit,*arrays[1:])
                returned=.5*np.sum(fit_residual**2+rho**2)
                stats["returned_storage"]=float(returned)
                stats["fit_storage_jump"]=float(returned-stats["energy_after"])
                budget=c["fit_storage_allowance"]*max(stats["energy_before"],1e-12)
                stats["fit_storage_budget"]=budget
                if method=="ph_sac":
                    stats["phase_gate_pass"]=(stats["inverse_gate_pass"] and
                        stats["midpoint_balance_defect"]<=1e-8*max(1.,stats["energy_before"]))
                    if c["require_solver_tolerance"]:
                        stats["phase_gate_pass"]=stats["phase_gate_pass"] and stats["solver_converged"]
                    stats["numerical_gate_pass"]=stats["phase_gate_pass"] and stats["fit_storage_jump"]<=budget
            complete_gate=stats["critic_fit_pass"] and stats["numerical_gate_pass"]
            stats["complete_numerical_gate_pass"]=complete_gate
            stats["fit_admission_mode"]=c["fit_admission"]
            admitted=(complete_gate or (finite and c["fit_admission"]=="empirical")) and stats["phase_gate_pass"]
        if not admitted:
            self.critics.load_state_dict(old_critics);self.critic_opt.load_state_dict(old_optimizer)
            stats.update(critic_admitted=False,actor_admitted=False,policy_gap_certificate=None)
            return stats
        self.critics.requires_grad_(False)
        actor_due=method!="ct_td3" or self.updates%c["td3_delay"]==0
        if actor_due:
            actions,lp=self.actor.sample(info,deterministic=method=="ct_td3")
            if method=="ct_td3":actor_loss=-self.q(info,actions)[0].mean()
            else:
                coefficient=c["alpha_reference"] if method in ("ct_sac","ct_sac_i") else self.alpha(batch.get("h_requested",batch["h"]))
                actor_loss=(coefficient*lp-self.q(info,actions).min(0).values).mean()
            self.actor_opt.zero_grad(set_to_none=True);actor_loss.backward();self.actor_opt.step()
            stats.update(actor_loss=float(actor_loss.detach()),actor_steps=1)
        self.critics.requires_grad_(True)
        if method!="ct_td3" or actor_due:
            with torch.no_grad():
                old_target=self.q(info,batch["action"],True)
                expected=(1-c["tau"])*old_target+c["tau"]*self.q(info,batch["action"])
                for a,b in zip(self.targets.parameters(),self.critics.parameters()):a.lerp_(b,c["tau"])
                for a,b in zip(self.target_actor.parameters(),self.actor.parameters()):a.lerp_(b,c["tau"])
                stats["target_prediction_mix_defect"]=float(abs(self.q(info,batch["action"],True)-expected).max())
        stats.update(critic_admitted=True,actor_admitted=actor_due,policy_gap_certificate=None)
        return stats

    def state(self):
        return dict(actor=self.actor.state_dict(),critics=self.critics.state_dict(),targets=self.targets.state_dict(),
                    target_actor=self.target_actor.state_dict(),actor_optimizer=self.actor_opt.state_dict(),
                    critic_optimizer=self.critic_opt.state_dict(),updates=self.updates)

    def load(self,s):
        for key in ("actor","critics","targets","target_actor"):getattr(self,key).load_state_dict(s[key])
        self.actor_opt.load_state_dict(s["actor_optimizer"]);self.critic_opt.load_state_dict(s["critic_optimizer"])
        self.updates=s["updates"]
