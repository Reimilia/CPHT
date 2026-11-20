"""Planar articulated Hamiltonian, learned positive inertial parameters, and FK.

All predictive computations use this Torch model. MuJoCo is used only to read
public robot geometry at construction and to acquire measured training labels.
"""
import numpy as np
import torch


def rotate(angle,point):
    c,s=torch.cos(angle),torch.sin(angle);x,z=point[...,0],point[...,1]
    return torch.stack((c*x+s*z,-s*x+c*z),-1)


def perpendicular(point):return torch.stack((point[...,1],-point[...,0]),-1)


def public_geometry(nominal):
    import mujoco
    m=nominal.model
    if (m.nq,m.nv,m.nbody)!=(9,9,8) or not np.allclose(m.body_quat[:,1:],0) or not np.allclose(m.jnt_pos,0):
        raise ValueError('Pure CPHT geometry currently supports the stock planar cheetah tree')
    if not np.allclose(m.jnt_axis[2:],[0,1,0]) or not np.allclose(m.jnt_axis[:2],[[1,0,0],[0,0,1]]):
        raise ValueError('Unexpected planar root/joint axes')
    inertias=[];points=[]
    for b in range(1,m.nbody):
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,m.body_iquat[b])
        inertias.append(float((rotation.reshape(3,3)[1]**2)@m.body_inertia[b]))
    for g in range(m.ngeom):
        body=int(m.geom_bodyid[g])
        if body==0:continue
        if int(m.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_CAPSULE):raise ValueError('Capsule geometry expected')
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[g]);axis=rotation.reshape(3,3)[:,2]
        for sign in (-1,1):
            point=m.geom_pos[g]+sign*m.geom_size[g,1]*axis
            points.append(dict(body=body,local=point[[0,2]].tolist(),radius=float(m.geom_size[g,0])))
    # Include torso, head, thighs and shins as well as feet in predicted contact.
    return dict(parents=m.body_parentid.tolist(),positions=m.body_pos[:,[0,2]].tolist(),
        com=m.body_ipos[:,[0,2]].tolist(),joint=m.body_jntadr.tolist(),
        masses=m.body_mass[1:].tolist(),inertias=inertias,armature=m.dof_armature[3:].tolist(),
        stiffness=m.jnt_stiffness[3:].tolist(),damping=m.dof_damping[3:].tolist(),
        spring_reference=m.qpos_spring.tolist(),limits=m.jnt_range[3:].tolist(),
        gears=m.actuator_gear[:,0].tolist(),gravity=m.opt.gravity[[0,2]].tolist(),points=points,
        xml_sha256=nominal.xml_sha256)


class Robot:
    """Known geometry and 32 positive physical coefficients (7+7+6+6+6)."""
    def __init__(self,geometry,parameters=None,device='cpu',dtype=torch.float32):
        self.geometry=geometry;self.device=device;self.dtype=dtype
        self.t=lambda x:torch.as_tensor(x,device=device,dtype=dtype)
        self.positions=self.t(geometry['positions']);self.com_local=self.t(geometry['com'])
        self.gravity=self.t(geometry['gravity']);self.limits=self.t(geometry['limits']);self.reference=self.t(geometry['spring_reference'])
        self.gears=self.t(geometry['gears']);self.B=torch.cat((torch.zeros((3,6),device=device,dtype=dtype),torch.diag(self.gears)),0)
        self.point_body=[p['body'] for p in geometry['points']]
        self.point_local=self.t([p['local'] for p in geometry['points']]);self.radius=self.t([p['radius'] for p in geometry['points']])
        prior=sum([geometry[k] for k in ('masses','inertias','armature','stiffness','damping')],[])
        self.parameters=self.t(prior if parameters is None else parameters)
        if self.parameters.shape!=(32,) or not bool(torch.isfinite(self.parameters).all()) or bool((self.parameters<=0).any()):
            raise ValueError('Expected 32 finite positive planar mechanical parameters')
        self.mass,self.inertia,self.armature,self.stiffness,self.damping=torch.split(self.parameters,[7,7,6,6,6])
        ancestors=np.zeros((8,9))
        for b in range(1,8):
            ancestors[b]=ancestors[geometry['parents'][b]]
            if b==1:ancestors[b,2]=1.
            else:ancestors[b,geometry['joint'][b]]=1.
        self.ancestors=self.t(ancestors)
        self.queries=0

    def kinematics(self,q,v):
        batch=q.shape[0];zero=torch.zeros((batch,2),device=q.device,dtype=q.dtype)
        origin=[zero];angle=[torch.zeros(batch,device=q.device,dtype=q.dtype)];velocity=[zero];acceleration=[zero]
        omega=[angle[0]];joint_origins=[None,None,None]+[None]*6
        for b in range(1,8):
            parent=self.geometry['parents'][b]
            offset=rotate(angle[parent],self.positions[b])
            if b==1:
                root=torch.stack((q[:,0],q[:,1]),-1)+offset
                origin.append(root);angle.append(q[:,2]);velocity.append(v[:,:2]);acceleration.append(zero);omega.append(v[:,2])
                joint_origins[2]=root
            else:
                j=self.geometry['joint'][b]
                origin.append(origin[parent]+offset);angle.append(angle[parent]+q[:,j])
                velocity.append(velocity[parent]+omega[parent][:,None]*perpendicular(offset))
                acceleration.append(acceleration[parent]-omega[parent][:,None]**2*offset)
                omega.append(omega[parent]+v[:,j]);joint_origins[j]=origin[b]
        origins=torch.stack(origin,1);angles=torch.stack(angle,1);omegas=torch.stack(omega,1)
        accelerations=torch.stack(acceleration,1)
        hinges=torch.stack([zero if p is None else p for p in joint_origins],1)
        def points(bodies,local):
            off=rotate(angles[:,bodies],local);pos=origins[:,bodies]+off
            delta=pos[:,:,None,:]-hinges[:,None,:,:]
            rotational=perpendicular(delta).permute(0,1,3,2)*self.ancestors[bodies][None,:,None,:]
            # Translational columns are constant for every robot body.
            translation=torch.zeros_like(rotational)
            translation[:,:,0,0]=1.;translation[:,:,1,1]=1.
            jac=rotational+translation
            bias=accelerations[:,bodies]-omegas[:,bodies,None]**2*off
            return pos,jac,bias
        com,jac,com_bias=points(list(range(1,8)),self.com_local[1:])
        contacts,cjac,cbias=points(self.point_body,self.point_local)
        return dict(com=com,J=jac,bias=com_bias,points=contacts,point_J=cjac,point_bias=cbias,
            origins=origins,angles=angles,hinges=hinges,omega=omegas,origin_bias=accelerations)

    def quantities(self,q,v):
        self.queries+=q.shape[0]
        k=self.kinematics(q,v);jac=k['J'];angular=self.ancestors[1:]
        M=torch.einsum('nbdi,b,nbdj->nij',jac,self.mass,jac)+torch.einsum('bi,b,bj->ij',angular,self.inertia,angular)[None]
        diagonal=torch.cat((torch.zeros(3,device=q.device,dtype=q.dtype),self.armature))
        M=M+torch.diag(diagonal)[None]
        bias=torch.einsum('nbdi,b,nbd->ni',jac,self.mass,k['bias']-self.gravity)
        passive=torch.cat((torch.zeros((len(q),3),device=q.device,dtype=q.dtype),
            self.stiffness*(q[:,3:]-self.reference[3:])+self.damping*v[:,3:]),1)
        com_j=torch.einsum('nbdi,b->ndi',jac,self.mass)/self.mass.sum()
        com=torch.einsum('nbd,b->nd',k['com'],self.mass)/self.mass.sum()
        momentum=torch.einsum('nij,nj->ni',M,v)
        kinetic=.5*(momentum*v).sum(1)
        potential=-(k['com']*self.gravity).sum(2)@self.mass+.5*(self.stiffness*(q[:,3:]-self.reference[3:])**2).sum(1)
        return dict(k,M=M,c=bias+passive,bias_force=bias,p=momentum,H=kinetic+potential,
            kinetic=kinetic,potential=potential,com_position=com,com_velocity=torch.einsum('ndi,ni->nd',com_j,v),
            dissipation=(self.damping*v[:,3:]**2).sum(1))

    def regressor(self,q,v,acceleration):
        k=self.kinematics(q,v);jac=k['J'];angular=self.ancestors[1:]
        point_acc=torch.einsum('nbdi,ni->nbd',jac,acceleration)+k['bias']-self.gravity
        mass=(jac*point_acc[:,:,:,None]).sum(2).transpose(1,2)
        angular_acc=torch.einsum('bi,ni->nb',angular,acceleration)
        inertia=angular.T[None]*angular_acc[:,None,:]
        def joint_columns(value):
            return torch.cat((torch.zeros((len(q),3,6),device=q.device,dtype=q.dtype),torch.diag_embed(value)),1)
        return torch.cat((mass,inertia,joint_columns(acceleration[:,3:]),
            joint_columns(q[:,3:]-self.reference[3:]),joint_columns(v[:,3:])),2)

    def acceleration(self,q,v,u,port):
        d=self.quantities(q,v)
        return torch.linalg.solve(d['M'],(u@self.B.T+port-d['c']).unsqueeze(-1)).squeeze(-1)

    def state(self):return dict(geometry=self.geometry,parameters=self.parameters.detach().cpu().tolist())
