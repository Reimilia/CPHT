"""A small batched primal active-set solver for the six bounded motor inputs."""
import torch


def bounded_quadratic(Q,rhs,iterations=16):
    action=torch.linalg.solve(Q,rhs[:,:,None]).squeeze(-1).clamp(-1,1)
    working=(action>=1).to(Q.dtype)-(action<=-1).to(Q.dtype)
    n=rhs.shape[1];identity=torch.eye(n,device=Q.device,dtype=Q.dtype)[None]
    count=0
    for _ in range(iterations):
        free=(working==0).to(Q.dtype)
        matrix=Q*free[:,:,None]*free[:,None,:]+identity*(1-free)[:,:,None]
        target=(rhs-(Q@working[:,:,None]).squeeze(-1))*free+working
        minimizer=torch.linalg.solve(matrix,target[:,:,None]).squeeze(-1)
        direction=(minimizer-action)*free
        stationary=direction.abs().max(1).values<1e-6
        safe=torch.where(direction.abs()>1e-9,direction,torch.ones_like(direction))
        distance=torch.where(direction>0,1-action,-1-action)/safe
        distance=torch.where((direction.abs()>1e-9)&(free>0),distance,torch.full_like(distance,float('inf')))
        step,blocking=distance.min(1);step=step.clamp(0,1)
        action=(action+step[:,None]*direction).clamp(-1,1)
        gradient=(Q@action[:,:,None]).squeeze(-1)-rhs
        violation=working*gradient
        largest,release=violation.max(1)
        release_mask=stationary&(largest>1e-5)
        block_mask=(~stationary)&(step<1-1e-6)
        replacement=working.gather(1,blocking[:,None]).squeeze(1)
        replacement=torch.where(block_mask,direction.gather(1,blocking[:,None]).squeeze(1).sign(),replacement)
        working=working.scatter(1,blocking[:,None],replacement[:,None])
        release_value=working.gather(1,release[:,None]).squeeze(1)
        working=working.scatter(1,release[:,None],torch.where(release_mask,torch.zeros_like(release_value),release_value)[:,None])
        count+=1
        if Q.device.type=='cpu' and bool((stationary&(~release_mask)).all()):break
    residual=action-(action+rhs-(Q@action[:,:,None]).squeeze(-1)).clamp(-1,1)
    return action,residual.abs().max(1).values,count
