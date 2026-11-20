"""Common causal information map and positive operator model for branch Op."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from bench.learner import mlp
from .config import observation_dim,is_path_task


def tensors(batch,device):
    return {k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in batch.items()}


class InformationMap(nn.Module):
    def __init__(self,c,mean,scale):
        super().__init__()
        self.c=c
        self.obs_dim=observation_dim(c)
        self.kind="identity" if c["case"]=="F" and c["sensor"]=="native" and c["experiment"]!="contact" and not is_path_task(c) else "causal_gru"
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32).clamp_min(.05))
        self.gru=nn.GRU(self.obs_dim+8,c["encoder_dim"],batch_first=True) if self.kind=="causal_gru" else None
        self.feature_dim=(self.obs_dim if self.gru is None else c["encoder_dim"])+2+int(c["boundary"]=="finite_horizon")

    def forward(self,window,g,h,time=None):
        if not self.c["use_reward_history"]:
            window=window.clone();window[:,:,self.obs_dim+6]=0
        normalized=(window-self.mean)/self.scale
        if self.gru is None:
            representation=normalized[:,-1,:self.obs_dim]
        else:
            representation=self.gru(normalized)[0][:,-1]
        pieces=[representation,g/10.,h/self.c["reference_time"]]
        if self.c["boundary"]=="finite_horizon":
            if time is None:raise ValueError("Finite-horizon information requires its clock")
            pieces.append((1-time/self.c["horizon"]).clamp(0,1))
        return torch.cat(pieces,dim=-1)

    def encode_batch(self,batch,next_state=False):
        return self(batch["next_window" if next_state else "window"],batch["g"],
                    batch["next_h"] if next_state else batch.get("h_requested",batch["h"]),batch["next_time" if next_state else "time"])


class ObservationPredictor(nn.Module):
    def __init__(self,information):
        super().__init__()
        self.information=information
        self.net=mlp(information.feature_dim+6,2*(information.obs_dim+1),[128,128])

    def forward(self,batch,short=False):
        window=batch["window"]
        if short:window=window[:,-max(1,window.shape[1]//2):]
        info=self.information(window,batch["g"],batch.get("h_requested",batch["h"]),batch["time"])
        mean,logvar=self.net(torch.cat([info,batch["action"]],1)).chunk(2,1)
        return mean,logvar.clamp(-6,4)

    def loss(self,batch):
        target=torch.cat([(batch["next_obs"]-self.information.mean[:self.information.obs_dim])/
                           self.information.scale[:self.information.obs_dim],batch["r_src"]],1)
        mean,logvar=self(batch)
        nll=.5*((target-mean).square()*torch.exp(-logvar)+logvar).mean()
        short,short_logvar=self(batch,short=True)
        # This is a predictive consistency proxy, not the posterior KL in (72).
        closure=.5*((logvar.detach()-short_logvar).exp()+(short-mean.detach()).square()*
                     torch.exp(-short_logvar)-1+short_logvar-logvar.detach()).mean()
        return nll+self.information.c["closure_weight"]*closure,dict(observation_nll=float(nll.detach()),
                                                                  predictive_closure_proxy=float(closure.detach()))


class OperatorModel(nn.Module):
    def __init__(self,feature_dim,nodes,c,reward_scale):
        super().__init__()
        self.c=c
        self.with_survival=is_path_task(c)
        self.members=nn.ModuleList([mlp(feature_dim+6,len(nodes["g"])+2+int(self.with_survival),[128,128]) for _ in range(c["ensembles"])])
        self.register_buffer("node_g",torch.as_tensor(nodes["g"],dtype=torch.float32).reshape(-1))
        self.register_buffer("reward_scale",torch.as_tensor(reward_scale,dtype=torch.float32).clamp_min(.01))
        self.version=0

    def predict(self,information,actions,g,member=None):
        p,r,_=self.predict_with_survival(information,actions,g,member)
        return p,r

    def predict_with_survival(self,information,actions,g,member=None):
        allowed=torch.isclose(g.reshape(-1,1),self.node_g.reshape(1,-1),atol=1e-5)
        if not bool(allowed.any(1).all()):
            raise ValueError("No same-task anchors. This finite operator supplies no held-out-task certificate.")
        networks=list(self.members) if member is None else [self.members[member]]
        predictions=[net(torch.cat([information,actions],1)) for net in networks]
        bank=torch.stack([torch.softmax(p[:,:len(self.node_g)].masked_fill(~allowed,-torch.inf),1) for p in predictions])
        survival=torch.stack([torch.sigmoid(p[:,-1:]) if self.with_survival else torch.ones_like(p[:,-1:]) for p in predictions])
        mean_survival=survival.mean(0)
        # Average the joint live-transition kernels. Averaging P and survival
        # independently would omit their ensemble dependence.
        weighted=(bank*survival).mean(0)
        probabilities=torch.where(mean_survival>1e-12,weighted/mean_survival.clamp_min(1e-12),bank.mean(0))
        reward=torch.stack([p[:,len(self.node_g):len(self.node_g)+2]*self.reward_scale for p in predictions]).mean(0)
        return probabilities,reward,mean_survival

    def loss(self,information,actions,g,target_rows,rewards,gamma,probes,member,survival_target=None,raw_gamma=None):
        p,r,survival=self.predict_with_survival(information,actions,g,member)
        row_losses=-(target_rows*torch.log(p.clamp_min(1e-12))).sum(1)
        if self.with_survival:
            if survival_target is None or raw_gamma is None:raise ValueError("Path operator fitting requires terminal labels and base discount")
            live=survival_target.reshape(-1)
            classification=(row_losses*live).sum()/live.sum().clamp_min(1)
            terminal_loss=nn.functional.binary_cross_entropy(survival.clamp(1e-6,1-1e-6),survival_target)
        else:classification=row_losses.mean();terminal_loss=survival.sum()*0
        reward_loss=((r-rewards)/self.reward_scale).square().mean()
        predicted_backup=r[:,1:]+(raw_gamma*survival if self.with_survival else gamma)*(p@probes)
        measured_backup=rewards[:,1:]+gamma*(target_rows@probes)
        probe_loss=(predicted_backup-measured_backup).square().mean()
        loss=classification+reward_loss+self.c["probe_weight"]*probe_loss+terminal_loss
        return loss,dict(row_cross_entropy=float(classification.detach()),reward_mse=float(reward_loss.detach()),
                         terminal_probe_mse=float(probe_loss.detach()),survival_bce=float(terminal_loss.detach()))


def row_targets(next_information,g,node_information,node_g,c):
    allowed=torch.isclose(g.reshape(-1,1),node_g.reshape(1,-1),atol=1e-5)
    scale=node_information.std(0,unbiased=False).clamp_min(.1)
    distance=torch.cdist(next_information/scale,node_information/scale).square()/next_information.shape[1]
    if not bool(allowed.any(1).all()):raise ValueError("No anchor with the known task descriptor")
    rows=torch.softmax((-distance/c["kernel_temperature"]).masked_fill(~allowed,-torch.inf),1)
    nearest=distance.masked_fill(~allowed,torch.inf).min(1).values
    return rows,nearest


def make_nodes(replay,indices,c):
    rng=np.random.default_rng(c["seed"]+211)
    groups=np.unique(replay.arrays["g"][indices])
    selected=[]
    for j,g in enumerate(groups):
        count=c["nodes"]//len(groups)+int(j<c["nodes"]%len(groups))
        eligible=np.asarray(indices)[replay.arrays["g"][indices,0]==g]
        if len(eligible)<count:raise ValueError("Insufficient per-task anchor coverage")
        selected.extend(rng.choice(eligible,count,replace=False).tolist())
    return replay.windows(selected)
