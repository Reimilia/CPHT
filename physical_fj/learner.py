"""Conventional measured-transition SAC targets. No finite value inverse."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from bench.learner import Actor,mlp


class Replay:
    def __init__(self,capacity,dimension,seed):
        self.capacity=capacity;self.size=self.count=0;self.rng=np.random.default_rng(seed)
        self.arrays={key:np.zeros((capacity,width),np.float32) for key,width in dict(
            obs=dimension,next_obs=dimension,action=6,R=1,r_src=1,h=1,h_requested=1,next_h=1,done=1).items()}
    def add(self,**row):
        i=self.count%self.capacity
        for key in self.arrays:self.arrays[key][i]=row[key]
        self.count+=1;self.size=min(self.count,self.capacity)
    def sample(self,n,device):
        ix=self.rng.integers(self.size,size=n)
        return {key:torch.as_tensor(a[ix],device=device) for key,a in self.arrays.items()}
    def state(self):return dict(arrays={k:v[:self.size].copy() for k,v in self.arrays.items()},count=self.count,size=self.size,rng=self.rng.bit_generator.state)
    def restore(self,state):
        for key,a in state['arrays'].items():self.arrays[key][:len(a)]=a
        self.count,self.size=state['count'],state['size'];self.rng.bit_generator.state=state['rng']


class Learner:
    def __init__(self,dimension,s,c,method,device):
        self.s,self.c,self.method,self.device=s,c,method,device
        self.actor=Actor(dimension,6,s['hidden']).to(device)
        self.critics=nn.ModuleList([mlp(dimension+6,1,s['hidden']) for _ in range(2)]).to(device)
        self.targets=deepcopy(self.critics).requires_grad_(False)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=s['lr'])
        self.critic_opt=torch.optim.Adam(self.critics.parameters(),lr=s['lr'])
        self.updates=0
    def q(self,obs,action,target=False):
        x=torch.cat([obs,action],1)
        return torch.stack([net(x) for net in (self.targets if target else self.critics)])
    def act(self,obs,deterministic=False):
        with torch.no_grad():return self.actor.sample(torch.as_tensor(obs,device=self.device).reshape(1,-1),deterministic)[0][0].cpu().numpy()
    def update(self,b):
        c=self.c;ct=self.method.startswith('ct_');hbar=b['h']/c['reference_time']
        discount=torch.exp(np.log(c['gamma_reference'])*hbar);alpha=c['alpha_reference']
        with torch.no_grad():
            following,lp=self.actor.sample(b['next_obs'])
            if ct:
                action,logp=self.actor.sample(b['obs'])
                current=self.q(b['obs'],action,True).min(0).values-alpha*logp
                future=self.q(b['next_obs'],following,True).min(0).values-alpha*lp
                target=b['r_src']+(1-b['done'])*(current+(discount*future-current)/(hbar+1e-8))
                target=torch.where(b['done']>0,b['R'],target)
            else:
                future=self.q(b['next_obs'],following,True).min(0).values-alpha*b['next_h']/c['reference_time']*lp
                target=b['R']+discount*(1-b['done'])*future
        prediction=self.q(b['obs'],b['action']);loss=(prediction-target).square().mean((1,2)).sum()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite simple SAC critic loss; inspect observed rewards and Q scale')
        self.critic_opt.zero_grad(set_to_none=True);loss.backward();self.critic_opt.step()
        self.critics.requires_grad_(False)
        actions,logp=self.actor.sample(b['obs'])
        coefficient=alpha if ct else alpha*b['h_requested']/c['reference_time']
        actor_loss=(coefficient*logp-self.q(b['obs'],actions).min(0).values).mean()
        if not bool(torch.isfinite(actor_loss)):raise FloatingPointError('Non-finite simple SAC actor loss; inspect observed rewards and Q scale')
        self.actor_opt.zero_grad(set_to_none=True);actor_loss.backward();self.actor_opt.step()
        self.critics.requires_grad_(True)
        with torch.no_grad():
            for target_parameter,source_parameter in zip(self.targets.parameters(),self.critics.parameters()):target_parameter.lerp_(source_parameter,self.s['tau'])
        self.updates+=1
        return dict(critic_loss=float(loss.detach()),actor_loss=float(actor_loss.detach()),q_abs_max=float(prediction.detach().abs().max()),
            target_abs_max=float(target.abs().max()),terminal_target_error=float((abs(target-b['R'])*b['done']).max()),
            linear_value_solves=0,updates=self.updates)
    def state(self):
        return dict(actor=self.actor.state_dict(),critics=self.critics.state_dict(),targets=self.targets.state_dict(),
            actor_optimizer=self.actor_opt.state_dict(),critic_optimizer=self.critic_opt.state_dict(),updates=self.updates)
    def load(self,state):
        for key in ('actor','critics','targets'):getattr(self,key).load_state_dict(state[key])
        self.actor_opt.load_state_dict(state['actor_optimizer']);self.critic_opt.load_state_dict(state['critic_optimizer']);self.updates=state['updates']
