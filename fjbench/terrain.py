"""Heightfield courses that reproduce the sebale16/ct-rl terrain setup.

Reference: environment/cheetah_terrain.py at UPSTREAM_COMMIT. The adapter
retains this benchmark's measurements, task cost, clocks, and replay format.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np
from bench.protocol import digest
from .config import TERRAIN_REFERENCE_COMMIT

UPSTREAM_COMMIT=TERRAIN_REFERENCE_COMMIT
UPSTREAM_BLOB="e95e50779d6389d2f1bc19b5f10f7fd8a518efec"
UPSTREAM_URL=f"https://github.com/sebale16/ct-rl/blob/{UPSTREAM_COMMIT}/environment/cheetah_terrain.py"
COHORTS=("iid","validation","layout","task")


def _segments(rng,length,limits):
    sizes=[];total=0.
    while total<length:
        size=float(rng.uniform(*limits));sizes.append(size);total+=size
    return np.asarray(sizes)


def sample_map(c,seed,cohort="iid",progress=1.,difficulty=None):
    if cohort not in COHORTS:
        raise ValueError("Matched CT terrain supports iid, validation, layout, and task cohorts; use the mesh backend for material/parameter stress tests")
    if c["case"]=="F" and cohort not in ("iid","validation"):
        raise ValueError("F retains one fixed terrain course")
    if difficulty is not None and difficulty!=1.:
        raise ValueError("Matched CT terrain has full difficulty at every training stage")
    fixed=c["case"]=="F"
    bank="fixed" if fixed else {"validation":"validation","layout":"test"}.get(cohort,"train")
    # Each caller supplies an episode seed. A fixed F course uses course 1 by
    # default. J iid uses the source's RandomState reset-seed convention.
    map_seed=int(c["contact_fixed_seed"] if fixed else (int(seed)+(2**31 if cohort=="layout" else 0))%2**32)
    rng=np.random.RandomState(map_seed)
    dx=c["terrain_patch_length"];resolution=c["terrain_resolution"];start=c["terrain_course_start"]
    patches=int(math.ceil((c["terrain_course_length"]-start)/dx));ncol=int(round(dx/resolution))+1
    x=start+resolution*np.arange(patches*(ncol-1)+1)
    edges=start+dx*np.arange(patches+1)
    elevation="flat" if c["contact_profile"]=="friction" else c["terrain_elevation"]
    friction="uniform" if c["contact_profile"]=="elevation" else "patches"
    if c["terrain_ramp_in"]>0:
        blend=np.clip((x-c["terrain_start_flat"])/c["terrain_ramp_in"],0,1)
        blend=blend*blend*(3-2*blend)
    else:blend=(x>=c["terrain_start_flat"]).astype(float)
    band=c["terrain_max_elevation"]
    if elevation=="flat":z=np.zeros_like(x)
    elif elevation=="hills":
        from scipy.ndimage import gaussian_filter1d
        z=gaussian_filter1d(rng.standard_normal(x.size),c["terrain_hill_wavelength"]/(2*np.pi*resolution),mode="reflect")
        z*=blend
        z=z*(c["terrain_hill_height"]/max(np.max(np.abs(z)),1e-12))
    elif elevation=="slopes":
        sizes=_segments(rng,float(x[-1]-x[0]),c["terrain_slope_segment"])
        grades=rng.uniform(-c["terrain_max_grade"],c["terrain_max_grade"],sizes.size)
        knots=start+np.concatenate([[0.],np.cumsum(sizes)])
        heights=np.concatenate([[0.],np.cumsum(grades*sizes)])
        heights=band-np.abs(np.mod(heights+band,4*band)-2*band)
        z=np.clip(np.interp(x,knots,heights)*blend,-band,band)
    else:
        sizes=_segments(rng,float(x[-1]-x[0]),c["terrain_step_segment"])
        rises=rng.uniform(-c["terrain_step_height"],c["terrain_step_height"],sizes.size)
        heights=np.clip(np.cumsum(rises),-band,band)
        z=heights[np.minimum(np.searchsorted(start+np.cumsum(sizes),x),heights.size-1)]
        z=np.clip(z*blend,-band,band)
    mu=np.full(patches,.4)
    if friction=="patches":
        mu=np.exp(rng.uniform(np.log(c["terrain_friction_min"]),np.log(c["terrain_friction_max"]),patches))
        mu[edges[:-1]<c["terrain_start_flat"]]=.4
    result=dict(schema=2,backend="ct_terrain",case=c["case"],profile=c["contact_profile"],
        bank=bank,index=map_seed,map_seed=map_seed,cohort=cohort,difficulty=1.,
        x=x.tolist(),z=z.tolist(),patch_edges=edges.tolist(),mu=mu.tolist(),
        tau=[.02]*patches,zeta=[1.]*patches,half_width=.8,solimp=[.9,.95,.001,.5,2.],condim=3,priority=1,
        ncol=ncol,max_elevation=band,hfield_base=.5,elevation=elevation,friction=friction,
        friction_min=c["terrain_friction_min"],friction_max=c["terrain_friction_max"],
        upstream_commit=UPSTREAM_COMMIT,upstream_blob=UPSTREAM_BLOB,
        interpretation="CT terrain physical course; benchmark reward and sensor contract retained")
    result["map_sha256"]=digest({k:v for k,v in result.items() if k!="cohort"})
    validate_map(result)
    return result


def validate_map(m):
    x=np.asarray(m["x"],float);z=np.asarray(m["z"],float);edges=np.asarray(m["patch_edges"],float)
    n=len(edges)-1
    if n<1 or m["ncol"]<2 or len(x)!=n*(m["ncol"]-1)+1 or len(z)!=len(x):
        raise ValueError("Invalid heightfield sample count")
    arrays=[x,z,edges]+[np.asarray(m[k],float) for k in ("mu","tau","zeta")]
    if any(not np.isfinite(a).all() for a in arrays) or any(len(a)!=n for a in arrays[3:]):
        raise ValueError("Invalid terrain arrays")
    if not np.all(np.diff(x)>0) or not np.all(np.diff(edges)>0):raise ValueError("Terrain coordinates must increase")
    if not np.allclose(x[::m["ncol"]-1],edges,rtol=0,atol=1e-9):raise ValueError("Heightfield patch boundaries differ")
    if m["half_width"]!=.8 or m["condim"]!=3 or m["priority"]!=1 or m["hfield_base"]!=.5:
        raise ValueError("CT terrain geometry/contact constants differ from the source")
    if m["max_elevation"]<=0 or np.max(np.abs(z))>m["max_elevation"]+1e-10:
        raise ValueError("Heights exceed the heightfield band")
    if any(np.any(a<=0) for a in arrays[3:]):raise ValueError("Terrain contact coefficients must be positive")
    if m["map_sha256"]!=digest({k:v for k,v in m.items() if k not in ("cohort","map_sha256")}):
        raise ValueError("Terrain manifest differs from its hash")


def query(m,positions):
    x=np.asarray(m["x"]);z=np.asarray(m["z"]);p=np.asarray(positions,float)
    j=np.clip(np.searchsorted(x,p,side="right")-1,0,len(x)-2)
    i=np.clip(np.searchsorted(m["patch_edges"],p,side="right")-1,0,len(m["mu"])-1)
    slope=(z[j+1]-z[j])/(x[j+1]-x[j])
    return dict(height=np.interp(p,x,z),slope=np.where((p<x[0])|(p>x[-1]),0,slope),
        mu=np.asarray(m["mu"])[i],tau=np.asarray(m["tau"])[i],zeta=np.asarray(m["zeta"])[i],patch=i)


def build_xml(xml,m):
    validate_map(m);root=ET.fromstring(xml)
    world=root.find("worldbody");ground=world.find("geom[@name='ground']")
    if ground is None:raise ValueError("Stock cheetah ground is missing")
    index=list(world).index(ground);world.remove(ground)
    assets=root.find("asset")
    if assets is None:assets=ET.SubElement(root,"asset")
    for i,(a,b) in enumerate(zip(m["patch_edges"][:-1],m["patch_edges"][1:])):
        name=f"terrain_{i}"
        ET.SubElement(assets,"hfield",name=name,nrow="2",ncol=str(m["ncol"]),
            size=f"{(b-a)/2} .8 {2*m['max_elevation']} .5")
        geom=ET.Element("geom",name=name,type="hfield",hfield=name,pos=f"{(a+b)/2} 0 {-m['max_elevation']}",
            conaffinity="1",condim="3",priority="1",friction=".4 .1 .1",rgba=".45 .45 .45 1")
        world.insert(index+i,geom)
    return ET.tostring(root,encoding="unicode")


def apply_heightfields(physics,m):
    """Write normalized heights after compile, as in the source reset hook."""
    model=physics.model;data=(np.asarray(m["z"])+m["max_elevation"])/(2*m["max_elevation"])
    # Use the source distribution endpoints for colour, including start pads.
    lo,hi=np.log(m.get("friction_min",.1)),np.log(m.get("friction_max",1.))
    for i,mu in enumerate(m["mu"]):
        name=f"terrain_{i}";hid=model.name2id(name,"hfield");gid=model.name2id(name,"geom")
        adr=model.hfield_adr[hid];cols=data[i*(m["ncol"]-1):i*(m["ncol"]-1)+m["ncol"]]
        model.hfield_data[adr:adr+2*m["ncol"]]=np.tile(cols,2)
        model.geom_friction[gid]=(mu,.1,.1)
        t=(np.log(mu)-lo)/(hi-lo) if m["friction"]=="patches" and hi>lo else .5
        model.geom_rgba[gid,:3]=(1-t)*np.array([.65,.8,.95])+t*np.array([.35,.22,.1])
    if getattr(physics,"_contexts",None):
        from dm_control.mujoco.wrapper import mjbindings
        with physics.contexts.gl.make_current() as ctx:
            for i in range(len(m["mu"])):
                ctx.call(mjbindings.mjlib.mjr_uploadHField,model.ptr,physics.contexts.mujoco.ptr,model.name2id(f"terrain_{i}","hfield"))
