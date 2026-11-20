"""Known non-diagonal robot template, positive calibration, measured force ports."""
import hashlib
import numpy as np


def full_mass(model,data):
    import mujoco
    matrix=np.empty((model.nv,model.nv))
    try:mujoco.mj_fullM(model,data,matrix)  # MuJoCo 3.17+
    except TypeError:mujoco.mj_fullM(model,matrix,data.qM)
    return matrix


class NominalRobot:
    def __init__(self,s):
        import mujoco
        from dm_control.suite import cheetah
        xml,assets=cheetah.get_model_and_assets()
        self.xml_sha256=hashlib.sha256(xml.encode() if isinstance(xml,str) else xml).hexdigest()
        self.model=mujoco.MjModel.from_xml_string(xml,assets)
        m=self.model
        # A deliberately miscalibrated public XML template. No terrain map,
        # friction label, or true environment model is read by this template.
        m.body_mass[:]*=s['nominal_mass_scale'];m.body_inertia[:]*=s['nominal_mass_scale']
        m.dof_armature[:]*=s['nominal_mass_scale'];m.jnt_stiffness[:]*=s['nominal_stiffness_scale']
        m.tendon_stiffness[:]*=s['nominal_stiffness_scale']
        m.dof_damping[:]*=s['nominal_damping_scale']
        m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        if m.opt.density or m.opt.viscosity or m.ntendon or np.any(m.body_gravcomp):
            raise ValueError('This small template assumes joint damping and conservative joint/tendon springs')
        self.data=mujoco.MjData(m);mujoco.mj_setConst(m,self.data)
        self.queries=0

    def evaluate(self,q,v):
        import mujoco
        m,d=self.model,self.data
        d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=0.;mujoco.mj_forward(m,d)
        matrix=full_mass(m,d);damping=m.dof_damping*v
        conservative=d.qfrc_passive+damping
        mujoco.mj_energyPos(m,d)
        spring_energy=.5*float(np.sum(m.jnt_stiffness*(q-m.qpos_spring)**2))
        self.queries+=1
        return dict(M0=matrix,p0=matrix@v,T0=.5*float(v@matrix@v),Vg0=float(d.energy[0])-spring_energy,Vs0=spring_energy,
            bias0=d.qfrc_bias.copy(),spring0=-conservative,damping0=damping.copy(),
            dissipation0=float(v@damping))


def ph_features(raw,basis,coefficients):
    a,k,b=coefficients
    return np.concatenate([raw[:9],a*basis['p0'],raw[18:],
        [a*basis['T0'],a*basis['Vg0']+k*basis['Vs0'],basis['port_power'],b*basis['dissipation0']]]).astype(np.float32)


class PhysicalEnvironment:
    """Exact-state/force sensor experiment; both compared policies receive it."""
    def __init__(self,c,s,seed,episode=0,cohort='iid',clock=None,nominal=None,purpose='train',**reset):
        import mujoco
        from fjbench.environment import Environment
        self.core=Environment(c,seed,episode,cohort,clock,purpose=purpose,**reset)
        self.nominal=nominal or NominalRobot(s)
        # Settings saved before v3.8 have no key and were trained with the hold
        # disclosed; keep that interface for them. New settings always set it.
        self.hold_disclosure=bool(s.get('hold_disclosure',True))
        self.terrain_probes='sensing' not in s and bool(s.get('pure',{}).get('terrain_probes'))
        self.contact_labels=self.terrain_probes or ('sensing' in s and 'pure' in s)
        self.point_contact_labels=s.get('pure',{}).get('contact_supervision','point_labels')=='point_labels'
        self.scratch=mujoco.MjData(self.core.base._env.physics.model.ptr)
        self.sensing=None
        if 'sensing' in s:
            from environment_belief.runtime import Runtime
            self.sensing=Runtime(self.core,self.nominal,s,seed)
        self.sensor_solves=0;self.current=self.observe()

    def __getattr__(self,name):return getattr(self.core,name)

    def announced_hold(self):
        """The hold a causal policy may see before it acts.

        With hold_disclosure False (the default), the next hold is unannounced
        (paper Sec. 2.6) and policies see the nominal fixed hold. The realized
        hold is still recorded after the step in info['h'].
        """
        return self.core.hold() if self.hold_disclosure else self.core.config['fixed_hold']

    def snapshot(self,action=None):
        import mujoco
        p=self.core.base._env.physics;m=p.model.ptr;d=self.scratch
        mujoco.mj_copyData(d,m,p.data.ptr)
        if action is not None:d.ctrl[:]=action
        mujoco.mj_forward(m,d);self.sensor_solves+=1
        return d

    def observe(self):
        d=self.snapshot();q=d.qpos.copy();v=d.qvel.copy();port=d.qfrc_constraint.copy()
        c=self.core.config;t=self.core.base.cur_t;torso=self.core.base._env.physics.model.name2id('torso','body')
        raw=np.concatenate([q,v,port,[(self.core.path_task.goal-d.subtree_com[torso,0])/c['path_length'],
            max(0.,1-t/c['horizon']),self.core.g/10.,self.announced_hold()/c['reference_time']]]).astype(np.float32)
        basis=self.nominal.evaluate(q,v);basis['port_power']=float(v@port)
        result=dict(raw=raw,basis=basis)
        if self.sensing is not None:
            result.update(self.sensing.observe(raw))
        if self.terrain_probes:
            count=2*len(c['contact_offsets'])
            measured=np.asarray(self.core.observation[-count-1:-1]).reshape(2,-1).T
            origin=d.xpos[torso]
            result['scan']=np.column_stack((origin[0]+np.asarray(c['contact_offsets']),origin[2]+measured[:,0],measured[:,1]))
            result['raw']=np.concatenate((raw,measured.reshape(-1))).astype(np.float32)
        return result

    def step(self,action,calibrate=False):
        import mujoco
        action=np.asarray(action,dtype=np.float32)
        label=dict(samples=[],contacts=[]);original=self.core.base._point_force
        def instrument():
            result=original()
            d=self.snapshot(action);basis=self.nominal.evaluate(d.qpos,d.qvel);p=self.core.base._env.physics
            x=basis['M0']@d.qacc+basis['bias0']
            label['samples'].append(dict(X=np.stack([x,basis['spring0'],basis['damping0']],1),
                y=d.qfrc_actuator.copy()+d.qfrc_constraint.copy(),v=d.qvel.copy(),q=d.qpos.copy(),
                acceleration=d.qacc.copy()))
            if self.contact_labels:
                records=[];jp=np.zeros((3,9));jr=np.zeros_like(jp);cf=np.zeros(6)
                for i in range(d.ncon):
                    if not self.point_contact_labels:break
                    con=d.contact[i];ga=int(p.model.geom_bodyid[int(con.geom1)]);gb=int(p.model.geom_bodyid[int(con.geom2)])
                    if ga and gb:continue
                    body,sign=(gb,1.) if ga==0 else (ga,-1.)
                    if not body:continue
                    mujoco.mj_contactForce(p.model.ptr,d,i,cf)
                    frame=np.asarray(con.frame).reshape(3,3)
                    normal=sign*frame[0];force=sign*frame.T@cf[:3]
                    local=np.asarray(d.xmat[body]).reshape(3,3).T@(con.pos-d.xpos[body])
                    mujoco.mj_jac(p.model.ptr,d,jp,jr,con.pos,body)
                    records.append(dict(body=body,local=local[[0,2]].copy(),normal=normal[[0,2]].copy(),
                        force=force[[0,2]].copy(),gap=float(con.dist),jacobian=jp[[0,2]].copy()))
                label['samples'][-1].update(scan=self.current['scan'].copy(),contacts=records,
                    u=action.copy(),constraint_port=d.qfrc_constraint.copy())
            if not label['contacts']:
                examples=[];contact_tau=np.zeros(9);jac=np.zeros((3,9));jacr=np.zeros_like(jac);force=np.zeros(6)
                for i in range(d.ncon):
                    con=d.contact[i];a,b=int(con.geom1),int(con.geom2)
                    ga=int(p.model.geom_bodyid[a]);gb=int(p.model.geom_bodyid[b])
                    if ga and gb:continue
                    body,sign=(gb,1.) if ga==0 else (ga,-1.)
                    if body==0:continue
                    mujoco.mj_contactForce(p.model.ptr,d,i,force)
                    f=sign*np.asarray(con.frame).reshape(3,3).T@force[:3]
                    mujoco.mj_jac(p.model.ptr,d,jac,jacr,con.pos,body)
                    tau=jac.T@f;contact_tau+=tau
                    examples.append(dict(body=p.model.id2name(body,'body'),point=con.pos.tolist(),
                        jacobian=jac.copy().tolist(),force=f.tolist(),generalized_force=tau.tolist(),
                        nominal_acceleration_response=np.linalg.solve(basis['M0'],tau).tolist()))
                label['contacts']=examples
                label['other_constraint_port']=(d.qfrc_constraint-contact_tau).tolist()
                if examples:
                    label['contact_time']=float(d.time)
                    label['contact_q']=d.qpos.copy().tolist()
            return result
        if calibrate:self.core.base._point_force=instrument
        try:_,info=self.core.step(action)
        finally:
            if calibrate:self.core.base._point_force=original
        self.current=self.observe()
        return self.current,info,label

    def close(self):self.core.close()
