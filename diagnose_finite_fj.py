"""Inspect one frozen critic-value recovery block. Perform zero training updates."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from bench.protocol import write_json
from fjbench.finite import solve,ResidualRecoveryError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint-file');group.add_argument('--reproduce',action='store_true')
    p.add_argument('--out',default='finite_diagnostics.json');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if a.reproduce:
        from fjbench.config import make_config
        c=make_config();n,k=8,2
        arrays=[np.zeros((2,n,k)),np.full((n,k),1e5),np.full((n,k),1-1e-4),np.full((n,k,n),1/n),np.zeros((n,k))]
        reference=dict(kind='large-value floating-point stress fixture; no trained policy',checkpoint=None)
    else:
        import torch
        from fjbench.calibration import load_prepared
        from fjbench.agent import Agent
        torch.manual_seed(a.seed);torch.set_num_threads(1)
        saved=torch.load(a.checkpoint_file,map_location='cpu',weights_only=False);c=saved['config']
        _,information,model,nodes,node_information=load_prepared(saved['public'],c,'cpu')
        agent=Agent(c,saved['method'],information,model,nodes,node_information,'cpu');agent.load(saved['agent'])
        with torch.no_grad():block=agent.finite_block()
        arrays=[block[key].detach().cpu().double().numpy() for key in ('initial','r','gamma','p','entropy')]
        arrays[3]/=arrays[3].sum(-1,keepdims=True)
        reference=dict(kind='one checkpoint block with frozen policy/model; fresh action quadrature',
            checkpoint=str(a.checkpoint_file),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint_file).read_bytes()).hexdigest())
    q,r,gamma,probability,ent=arrays
    matrix=np.eye(len(probability))-(gamma[...,None]*probability).mean(1)
    result=dict(reference,training_updates=0,q_abs_max=float(abs(q).max()),reward_abs_max=float(abs(r).max()),
        max_discount=float(gamma.max()),contraction_gap=float(1-gamma.max()),linear_condition_2=float(np.linalg.cond(matrix)))
    for name,config in (('absolute_only',dict(c,inverse_relative_tolerance=0.,inverse_refinement_steps=0)),
                        ('scale_aware',dict(c,inverse_relative_tolerance=1e-12,inverse_refinement_steps=2))):
        try:
            _,_,stats=solve(*arrays,config)
            result[name]=dict(passed=True,diagnostics=stats)
        except ResidualRecoveryError as error:result[name]=dict(passed=False,diagnostics=error.diagnostics)
    write_json(a.out,result)
    print(f"Wrote frozen-block diagnostics to {a.out}; no training updates.")


if __name__=='__main__':main()
