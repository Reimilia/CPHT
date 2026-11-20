"""Measured local terrain and a learned, finite-iteration contact QP closure."""
import torch
from .mechanics import perpendicular


def terrain(scan,x):
    """Linear interpolation of the latest measured scan. No environment access."""
    if scan.ndim==2:scan=scan[None].expand(x.shape[0],-1,-1)
    grid=scan[:,:,0].contiguous();index=torch.searchsorted(grid,x.contiguous()).sub(1).clamp(0,grid.shape[1]-2)
    left=scan.gather(1,index[:,:,None].expand(-1,-1,3));right=scan.gather(1,(index+1)[:,:,None].expand(-1,-1,3))
    fraction=((x-left[:,:,0])/(right[:,:,0]-left[:,:,0])).clamp(0,1)
    h=left[:,:,1]+fraction*(right[:,:,1]-left[:,:,1])
    slope=(left[:,:,2]+fraction*(right[:,:,2]-left[:,:,2])).clamp(-1.5,1.5)
    # Extrapolation uses the endpoint tangent and is flagged by the caller.
    h=h+torch.where(x<grid[:,:1],(x-grid[:,:1])*scan[:,:1,2],torch.zeros_like(x))
    h=h+torch.where(x>grid[:,-1:],(x-grid[:,-1:])*scan[:,-1:,2],torch.zeros_like(x))
    return h,slope


def cone_projection(force,mu):
    n,t=force[...,0],force[...,1];a=t.abs()
    inside=(n>=0)&(a<=mu*n);zero=n+mu*a<=0
    projected=(n+mu*a)/(1+mu*mu)
    nn=torch.where(inside,n,torch.where(zero,torch.zeros_like(n),projected))
    tt=torch.where(inside,t,torch.where(zero,torch.zeros_like(t),mu*projected*t.sign()))
    return torch.stack((nn,tt),-1)


class Contact:
    def __init__(self,robot,parameters,options):
        self.robot=robot;self.parameters=robot.t(parameters);self.options=options
        if self.parameters.shape!=(4,) or bool((self.parameters<=0).any()):raise ValueError('Positive contact k,b,r,mu required')
        self.queries=0

    def solve(self,q,v,u,scan,quantities=None,force_correction=None,elapsed=0.,friction=None):
        robot=self.robot;d=quantities if quantities is not None else robot.quantities(q,v)
        mu=self.parameters[3] if friction is None else friction
        p,jac,bias=d['points'],d['point_J'],d['point_bias']
        h,slope=terrain(scan,p[:,:,0]);normal=torch.stack((-slope,torch.ones_like(slope)),-1)
        normal=normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True)
        gap=(p[:,:,1]-h)*normal[:,:,1]-robot.radius[None]
        # Contact material points include the capsule radius lever arm.
        radius_offset=-robot.radius[None,:,None]*normal
        ancestors=robot.ancestors[robot.point_body]
        jac=jac+perpendicular(radius_offset)[:,:,:,None]*ancestors[None,:,None,:]
        bias=bias-d['omega'][:,robot.point_body,None]**2*radius_offset
        tangent=perpendicular(normal);frame=torch.stack((normal,tangent),2)
        count=min(self.options['max_contacts'],p.shape[1]);indices=torch.topk(-gap,count,dim=1).indices
        selected=gap.gather(1,indices)
        def gather(a):return a.gather(1,indices.reshape(indices.shape+(1,)*(a.ndim-2)).expand(indices.shape+a.shape[2:]))
        frame,jac,bias=gather(frame),gather(jac),gather(bias)
        active=(selected<self.options['contact_margin']).to(q.dtype)
        J=torch.einsum('npad,npdi->npai',frame,jac)*active[:,:,None,None]
        J=J.reshape(len(q),2*count,9)
        Jbias=(frame*bias[:,:,None,:]).sum(-1).reshape(len(q),2*count)*active.repeat_interleave(2,1)
        free=torch.linalg.solve(d['M'],(u@robot.B.T-d['c']).unsqueeze(-1)).squeeze(-1)
        response=torch.linalg.solve(d['M'],J.transpose(1,2))
        raw_W=J@response
        regularizer=self.parameters[2]*torch.diagonal(raw_W,dim1=1,dim2=2).clamp_min(1e-5)+1e-5
        W=raw_W+torch.diag_embed(regularizer)
        contact_velocity=(J@v[:,:,None]).reshape(len(q),count,2)
        desired=torch.stack((-self.parameters[0]*selected-self.parameters[1]*contact_velocity[:,:,0],
            -self.parameters[1]*contact_velocity[:,:,1]),-1)*active[:,:,None]
        rhs=desired.reshape(len(q),2*count)-(J@free[:,:,None]).squeeze(-1)-Jbias
        L=W.abs().sum(-1).max(-1).values.clamp_min(1e-6)
        force=(cone_projection(torch.linalg.solve(W,rhs[:,:,None]).reshape(len(q),count,2),mu)*active[:,:,None]).reshape(len(q),2*count)
        extrapolated=force;momentum=1.
        for _ in range(self.options['contact_iterations']):
            proposal=extrapolated+(rhs-(W@extrapolated[:,:,None]).squeeze(-1))/L[:,None]
            following=(cone_projection(proposal.reshape(len(q),count,2),mu)*active[:,:,None]).reshape(len(q),2*count)
            next_momentum=(1+(1+4*momentum**2)**.5)/2
            extrapolated=following+(momentum-1)/next_momentum*(following-force)
            force=following;momentum=next_momentum
        residual=force-cone_projection((force+rhs-(W@force[:,:,None]).squeeze(-1)).reshape(len(q),count,2),mu).reshape_as(force)
        # Dimensionless residual: use the actual 1/L projected iteration, then
        # normalize its force change by max(1 N, current force infinity norm).
        scaled_proposal=force+(rhs-(W@force[:,:,None]).squeeze(-1))/L[:,None]
        scaled_residual=force-cone_projection(scaled_proposal.reshape(len(q),count,2),mu).reshape_as(force)
        relative_residual=scaled_residual.abs().max(1).values/force.abs().max(1).values.clamp_min(1.)
        if force_correction is not None:
            full=force_correction
            if full.ndim==2:full=full[None].expand(len(q),-1,-1)
            correction=full.gather(1,indices[:,:,None].expand(-1,-1,2))
            fade=torch.exp(robot.t(-elapsed/self.options['observer_decay']))
            force=(cone_projection(force.reshape(len(q),count,2)+fade*correction,mu)*active[:,:,None]).reshape_as(force)
        tau=(J.transpose(1,2)@force[:,:,None]).squeeze(-1)
        # Joint-stop reactions are another explicit environment port.
        below=(robot.limits[:,0]-q[:,3:]).clamp_min(0);above=(q[:,3:]-robot.limits[:,1]).clamp_min(0)
        stop=self.options['limit_stiffness']*(below-above)
        stop=stop-self.options['limit_damping']*torch.where((below>0)|(above>0),v[:,3:],torch.zeros_like(v[:,3:]))
        tau=tau+torch.cat((torch.zeros_like(q[:,:3]),stop),1)
        self.queries+=len(q)
        return dict(port=tau,normal_force=force.reshape(len(q),count,2)[:,:,0],
            tangent_force=force.reshape(len(q),count,2)[:,:,1],gap=selected,
            projected_residual=residual.abs().max(1).values,relative_projected_residual=relative_residual,
            J=J,force=force,indices=indices,active=active)

    def observe(self,q,v,u,scan,measured):
        """Estimate a causal point-force correction from the measured port."""
        result=self.solve(q,v,u,scan);J=result['J'];initial=result['force'];count=result['active'].shape[1]
        ridge=self.options['observer_ridge'];Q=J@J.transpose(1,2)+ridge*torch.eye(2*count,device=q.device)[None]
        # Remove predicted joint-stop reactions from the observed contact target.
        target=measured-result['port']+(J.transpose(1,2)@initial[:,:,None]).squeeze(-1)
        rhs=(J@target[:,:,None]).squeeze(-1)+ridge*initial
        L=Q.abs().sum(-1).max(-1).values.clamp_min(1e-5);force=initial
        for _ in range(20):
            proposal=force+(rhs-(Q@force[:,:,None]).squeeze(-1))/L[:,None]
            force=(cone_projection(proposal.reshape(len(q),count,2),self.parameters[3])*result['active'][:,:,None]).reshape_as(force)
        before=(measured-result['port']).square().mean(1)
        residual=(J.transpose(1,2)@(force-initial)[:,:,None]).squeeze(-1)
        after=(measured-result['port']-residual).square().mean(1)
        delta=torch.where((after<=before)[:,None],force-initial,torch.zeros_like(force))
        full=torch.zeros((len(q),len(self.robot.point_body),2),device=q.device,dtype=q.dtype)
        full.scatter_(1,result['indices'][:,:,None].expand(-1,-1,2),delta.reshape(len(q),count,2))
        return full,dict(observer_error_before=before.sqrt(),observer_error_after=torch.minimum(before,after).sqrt())
