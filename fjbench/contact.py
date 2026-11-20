"""Reproducible terrain maps, permitted sensors, and compiled contact surfaces.

Map truth is private. Only ``measure`` supplies the declared actor sensor.
The map is piecewise linear and extruded along y. Every wedge has a shared
top edge with its neighbours; the old plane has no collision role.
"""
from bisect import bisect_right
from copy import deepcopy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from bench.protocol import digest

CONTACT_COHORTS=("iid","validation","layout","terrain","friction","material","contact",
                 "regime","task","joint")


def curriculum_level(c,progress=1.):
    if c["case"]=="F" or not c["contact_curriculum"]:return 1.
    if not 0<=progress<=1:raise ValueError("Training progress must lie in [0,1]")
    i=bisect_right(c["contact_stage_boundaries"],progress)-1
    return float(c["contact_stage_levels"][i])


def sample_map(c,seed=0,cohort="iid",progress=1.,difficulty=None):
    if c.get("contact_backend","mesh")=="ct_terrain":
        from .terrain import sample_map as sample_terrain
        return sample_terrain(c,seed,cohort,progress,difficulty)
    if cohort not in CONTACT_COHORTS:raise ValueError(f"Contact cohort {cohort}")
    if c["case"]=="F" and cohort not in ("iid","validation"):raise ValueError("F uses one fixed map")
    if cohort=="terrain" and c["contact_profile"] in ("friction","material"):
        raise ValueError("Terrain transfer needs elevation or mixed geometry")
    if cohort=="friction" and c["contact_profile"]=="elevation":
        raise ValueError("Friction transfer needs a spatial friction profile")
    if cohort=="material" and c["contact_profile"] in ("elevation","friction"):
        raise ValueError("Material transfer needs variable contact response")
    level=curriculum_level(c,progress) if cohort=="iid" else 1.
    if difficulty is not None:
        if not 0<difficulty<=1:raise ValueError("Difficulty must lie in (0,1]")
        level=1. if c["case"]=="F" else float(difficulty)
    bank="train"
    if cohort=="validation" and c["case"]=="J":bank="validation"
    elif cohort in ("layout","contact","joint"):bank="test"
    # Independent seed namespaces ensure disjoint layouts without consulting a
    # network input, training seed, hidden robot parameter, or task descriptor.
    select=np.random.default_rng(int(seed)+104729)
    index=0 if c["case"]=="F" else int(select.integers(c[f"contact_{bank}_maps"]))
    map_seed=int(c["contact_fixed_seed"]) if c["case"]=="F" else {"train":100_000,"validation":200_000,"test":300_000}[bank]+index
    rng=np.random.default_rng(map_seed)
    draw={name:float(rng.random()) for name in ("amplitude","wavelength","phase")}
    def value(name,test=False):
        lo,hi=c[f"contact_{name}_{'test' if test else 'train'}"]
        return lo+(hi-lo)*draw.get(name,.5)
    terrain_test=cohort in ("terrain","contact","joint")
    friction_test=cohort in ("friction","contact","joint")
    material_test=cohort in ("material","contact","joint")
    n=c["contact_patches"];dx=c["contact_patch_length"]
    x=np.arange(n+1,dtype=float)*dx-4*dx
    amp=value("amplitude",terrain_test)*level
    wavelength=value("wavelength",terrain_test)
    phase=2*np.pi*draw["phase"]
    # The initial pad is level. A C1 envelope joins the sampled landscape.
    envelope=np.clip((np.abs(x)-1.5)/2,0,1)
    envelope=envelope*envelope*(3-2*envelope)
    z=amp*envelope*(.7*np.sin(2*np.pi*x/wavelength+phase)+.3*np.sin(2*np.pi*x/(1.7*wavelength)-phase))
    if c["contact_profile"] in ("friction","material"):z[:]=0
    # Piecewise constant material blocks. All draws are consumed in every
    # cohort, so a single-factor intervention preserves the other fields.
    u=rng.random((n,3));blocks=np.minimum(np.arange(n)//2,n-1)
    u=u[blocks]
    def patch(name,column,test):
        lo,hi=c[f"contact_{name}_{'test' if test else 'train'}"]
        return lo+(hi-lo)*u[:,column]
    mu=patch("mu",0,friction_test)
    tau=patch("tau",1,material_test)
    zeta=patch("zeta",2,material_test)
    mu=1+level*(mu-1)
    tau=.02+level*(tau-.02)
    zeta=1+level*(zeta-1)
    if c["contact_profile"]=="elevation":mu[:]=.8
    if c["contact_profile"] in ("elevation","friction"):tau[:]=.02;zeta[:]=1.
    result=dict(schema=1,case=c["case"],profile=c["contact_profile"],bank=bank,index=index,
        map_seed=map_seed,cohort=cohort,difficulty=level,x=x.tolist(),z=z.tolist(),
        mu=mu.tolist(),tau=tau.tolist(),zeta=zeta.tolist(),half_width=c["contact_half_width"],
        solimp=[.9,.95,.001,.5,2.],condim=3,priority=2,
        interpretation="synthetic MuJoCo contact presets; no measured material calibration")
    result["map_sha256"]=digest({k:v for k,v in result.items() if k not in ("cohort",)})
    validate_map(result)
    return result


def validate_map(m):
    if m.get("backend")=="ct_terrain":
        from .terrain import validate_map as validate_terrain
        return validate_terrain(m)
    required={"x","z","mu","tau","zeta","half_width","priority","condim","solimp"}
    if not required<=m.keys():raise ValueError("Incomplete contact map")
    x=np.asarray(m["x"],float);z=np.asarray(m["z"],float);n=len(x)-1
    if n<1 or len(z)!=len(x) or not np.all(np.diff(x)>0):raise ValueError("Terrain vertices must increase")
    arrays=[x,z]+[np.asarray(m[k],float) for k in ("mu","tau","zeta")]
    if any(not np.isfinite(a).all() for a in arrays) or any(len(a)!=n for a in arrays[2:]):
        raise ValueError("Invalid patch arrays")
    if any(np.any(a<=0) for a in arrays[2:]) or m["half_width"]<=0:
        raise ValueError("Positive contact and geometry parameters required")
    if m["condim"]!=3 or m["priority"]!=2:raise ValueError("This benchmark fixes condim=3 and ground priority=2")
    if "map_sha256" in m and m["map_sha256"]!=digest({k:v for k,v in m.items() if k not in ("cohort","map_sha256")}):
        raise ValueError("Contact manifest content differs from its recorded hash")


def query(m,positions):
    """Exact piecewise-linear top, slope, and contact fields (private)."""
    if m.get("backend")=="ct_terrain":
        from .terrain import query as query_terrain
        return query_terrain(m,positions)
    x=np.asarray(m["x"]);z=np.asarray(m["z"]);p=np.asarray(positions,float)
    i=np.searchsorted(x,p,side="right")-1;i=np.clip(i,0,len(x)-2)
    slope=(z[i+1]-z[i])/(x[i+1]-x[i])
    height=z[i]+slope*(np.clip(p,x[0],x[-1])-x[i])
    # Endpoint pads are horizontal and extend the track outside its mesh bank.
    slope=np.where((p<x[0])|(p>x[-1]),0,slope)
    return dict(height=height,slope=slope,mu=np.asarray(m["mu"])[i],
                tau=np.asarray(m["tau"])[i],zeta=np.asarray(m["zeta"])[i],patch=i)


def measure(c,m,root_x,root_z,rng,friction_scale=1.):
    """One synchronous, zero-latency sample; no map ID or future measurement."""
    if c["contact_sensor"]=="history":return np.empty(0,np.float32)
    fields=query(m,root_x+np.asarray(c["contact_offsets"]))
    out=[]
    for name in ("height","slope")+( ("mu","tau","zeta") if c["contact_sensor"]=="material" else () ):
        val=fields[name].copy()
        if name=="height":val-=root_z
        if name=="mu":val*=friction_scale
        out.append(val+rng.normal(0,c[f"contact_{name}_std"],len(val)))
    return np.concatenate(out).astype(np.float32)


def build_xml(xml,m):
    """Extrude continuous top edges into convex, closed static mesh wedges."""
    if m.get("backend")=="ct_terrain":
        from .terrain import build_xml as build_terrain
        return build_terrain(xml,m)
    validate_map(m)
    root=ET.fromstring(xml)
    world=root.find("worldbody");assets=root.find("asset")
    if assets is None:assets=ET.SubElement(root,"asset")
    planes=[g for g in world.findall("geom") if g.get("type")=="plane"]
    for plane in planes:plane.set("contype","0");plane.set("conaffinity","0");plane.set("rgba","0 0 0 0")
    x=np.asarray(m["x"]);z=np.asarray(m["z"]);w=m["half_width"];bottom=float(z.min()-1)
    def add_surface(i,a,b,za,zb,material_index):
        name=f"contact_patch_{i}"
        vertices=[[a,-w,bottom],[a,w,bottom],[b,-w,bottom],[b,w,bottom],
                  [a,-w,za],[a,w,za],[b,-w,zb],[b,w,zb]]
        faces=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
               [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        ET.SubElement(assets,"mesh",name=name,vertex=" ".join(map(str,np.ravel(vertices))),
                      face=" ".join(map(str,np.ravel(faces))))
        j=material_index;mu=m["mu"][j];tau=m["tau"][j];damp=m["zeta"][j]
        # Colour is a preview/audit cue. The actor receives no RGB image.
        u=float(np.clip((mu-.1)/1.1,0,1));v=float(np.clip((tau-.02)/.06,0,1))
        rgba=[.18+.45*(1-u),.3+.4*u,.45+.25*v,1.]
        ET.SubElement(world,"geom",name=name,type="mesh",mesh=name,contype="1",conaffinity="1",
            condim="3",priority="2",friction=f"{mu} .1 .1",solref=f"{tau} {damp}",
            solimp=" ".join(map(str,m["solimp"])),rgba=" ".join(map(str,rgba)))
    for i in range(len(x)-1):add_surface(str(i),x[i],x[i+1],z[i],z[i+1],i)
    add_surface("left",x[0]-100,x[0],z[0],z[0],0)
    add_surface("right",x[-1],x[-1]+100,z[-1],z[-1],len(x)-2)
    return ET.tostring(root,encoding="unicode")


class ContactAudit:
    """Per-substep solver output, separate from actor/replay observations."""
    def __init__(self,physics,friction_scale=1.):
        import mujoco
        self.p=physics;self.scale=friction_scale
        self.data=mujoco.MjData(physics.model.ptr)
        self.ground={i for i in range(physics.model.ngeom) if (physics.model.id2name(i,"geom") or "").startswith(("contact_patch_","terrain_"))}
        self.jac=np.zeros((3,physics.model.nv));self.jacr=np.zeros_like(self.jac)
        self.force=np.zeros(6);self.reset()

    def reset(self):
        self.records=[];self.samples=0;self.work=0.;self.slip_integral=0.;self.max_penetration=0.
        self.pair_error=0.;self.normal_impulse=0.;self.back_impulse=0.;self.front_impulse=0.
        self.pairs_checked=0;self.tau_error=0.;self.zeta_error=0.

    def sample(self,dt):
        import mujoco
        p=self.p;m=p.model;d=self.data
        # dm_control can refresh contact geometry with mj_step1 after integrating
        # while leaving acceleration-stage forces from the preceding state.
        # Re-solve on a copy so each diagnostic force matches its point/Jacobian.
        # The simulated trajectory and its warm-start data remain unchanged.
        mujoco.mj_copyData(d,m.ptr,p.data.ptr)
        mujoco.mj_forward(m.ptr,d)
        records=[]
        for i in range(d.ncon):
            con=d.contact[i];a,b=int(con.geom1),int(con.geom2)
            if a not in self.ground and b not in self.ground:continue
            ground,robot=(a,b) if a in self.ground else (b,a)
            body=int(m.geom_bodyid[robot]);body_name=m.id2name(body,"body")
            mujoco.mj_contactForce(m.ptr,d,i,self.force)
            sign=1. if ground==a else -1.
            force=sign*np.asarray(con.frame).reshape(3,3).T@self.force[:3]
            mujoco.mj_jac(m.ptr,d,self.jac,self.jacr,np.asarray(con.pos),body)
            velocity=self.jac@d.qvel;frame=np.asarray(con.frame).reshape(3,3)
            tangent=velocity-frame[0]*(frame[0]@velocity)
            normal=float(max(0,self.force[0]));tforce=float(np.linalg.norm(self.force[1:3]))
            mu=float(con.friction[0]);expected=float(m.geom_friction[ground,0])
            self.pair_error=max(self.pair_error,abs(mu-expected))
            self.pairs_checked+=1
            self.tau_error=max(self.tau_error,abs(float(con.solref[0]-m.geom_solref[ground,0])))
            self.zeta_error=max(self.zeta_error,abs(float(con.solref[1]-m.geom_solref[ground,1])))
            penetration=float(max(0,-con.dist));self.max_penetration=max(self.max_penetration,penetration)
            speed=float(np.linalg.norm(tangent));power=float(force@velocity)
            self.work+=power*dt;self.slip_integral+=speed*dt;self.normal_impulse+=normal*dt
            if body_name=="bfoot":self.back_impulse+=normal*dt
            if body_name=="ffoot":self.front_impulse+=normal*dt
            tau=self.jac.T@force
            records.append(dict(body=body_name,ground=m.id2name(ground,"geom"),point=con.pos.tolist(),
                normal_force=normal,tangential_force=tforce,mu_effective=mu,mu_expected=expected,
                solref=con.solref.tolist(),solref_expected=m.geom_solref[ground].tolist(),
                friction_utilization=tforce/max(mu*normal,1e-9),slip_speed=speed,
                penetration=penetration,world_force=force.tolist(),generalized_force=tau.tolist(),
                directly_forced_dofs=np.flatnonzero(np.abs(tau)>1e-8).tolist()))
        self.samples+=1;self.records=records

    def summary(self):
        return dict(samples=self.samples,contact_point_work=self.work,
            audit_forward_solves=self.samples,
            contact_pairs_checked=self.pairs_checked,max_time_constant_error=self.tau_error,
            max_damping_ratio_error=self.zeta_error,
            contact_slip_integral=self.slip_integral,max_penetration=self.max_penetration,
            max_effective_friction_error=self.pair_error,normal_impulse=self.normal_impulse,
            back_foot_normal_impulse=self.back_impulse,front_foot_normal_impulse=self.front_impulse,
            last_contacts=deepcopy(self.records))


def plot_map(m,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.asarray(m["x"]);edges=np.asarray(m.get("patch_edges",m["x"]));centres=(edges[:-1]+edges[1:])/2
    fig,axs=plt.subplots(3,1,figsize=(11,6),sharex=True,layout="constrained")
    axs[0].plot(x,m["z"],color="#254e70");axs[0].set_ylabel("height [m]")
    axs[1].stairs(m["mu"],edges,color="#b75137");axs[1].set_ylabel("sliding coefficient")
    axs[2].step(centres,m["tau"],where="mid",label="time constant [s]")
    axs[2].set_ylabel("contact time [s]");axs[2].set_xlabel("world x [m]")
    for ax in axs:ax.grid(alpha=.2);ax.set_xlim(-2,min(32,x[-1]))
    fig.suptitle(f"{m['case']} / {m['profile']}   {m['bank']}:{m['index']}   difficulty {m['difficulty']:.2f}")
    fig.savefig(path,dpi=160);plt.close(fig)
