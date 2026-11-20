"""Compare physical courses with the pinned upstream module; perform no RL fit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def verify(upstream_file,out,fixtures=None):
    import numpy as np
    import mujoco
    from dm_control.suite import cheetah,common
    from fjbench.config import config_from
    from fjbench.terrain import UPSTREAM_COMMIT,UPSTREAM_BLOB,sample_map,build_xml,apply_heightfields
    from bench.protocol import ROOT,versions,write_json
    path=Path(upstream_file);content=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if blob!=UPSTREAM_BLOB:raise ValueError(f'Expected upstream blob {UPSTREAM_BLOB}; received {blob}')
    spec=importlib.util.spec_from_file_location('_checked_ct_terrain',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[];arrays={}
    for name in ('hills','friction','terrain','slopes','steps'):
        for seed in (1,7):
            c=config_from(ROOT/f'cases/path_speed_ct_{name}.json','J','contact')
            m=sample_map(c,seed)
            task=module.CheetahTerrain(random=seed,terrain_seed=None,terrain_obs='none',
                elevation=m['elevation'],friction=m['friction'])
            rng=task._course_rng();task.heights=task._draw_heights(rng);task.patch_friction=task._draw_friction(rng)
            expected=cheetah.Physics.from_xml_string(task.model_xml(),common.ASSETS)
            task._write_course(expected)
            xml,assets=cheetah.get_model_and_assets()
            actual=cheetah.Physics.from_xml_string(build_xml(xml,m),assets)
            apply_heightfields(actual,m)
            errors={}
            try:
                for field,reference in (('x',task.grid_x),('z',task.heights),('mu',task.patch_friction),('patch_edges',task.patch_edges)):
                    errors[field]=float(np.max(np.abs(np.asarray(m[field])-reference)))
                    np.testing.assert_array_equal(m[field],reference)
                    arrays[f'{name}_{seed}_{field}']=reference.copy()
                fields=('body_mass','body_inertia','dof_damping','actuator_gear','geom_type','geom_pos','geom_size',
                    'geom_contype','geom_conaffinity','geom_condim','geom_priority','geom_friction','geom_solref','geom_solimp',
                    'geom_rgba','hfield_size','hfield_nrow','hfield_ncol','hfield_data')
                for field in fields:
                    a=np.asarray(getattr(actual.model,field));b=np.asarray(getattr(expected.model,field))
                    errors[field]=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
                    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,err_msg=field)
                errors['gravity']=float(np.max(np.abs(actual.model.opt.gravity-expected.model.opt.gravity)))
                np.testing.assert_array_equal(actual.model.opt.gravity,expected.model.opt.gravity)
                assert actual.model.opt.integrator==expected.model.opt.integrator
                assert actual.model.opt.timestep==expected.model.opt.timestep
                ray_error=0.;geomid=np.zeros(1,np.int32)
                for x in np.linspace(task.course_start+.01,task.course_end-.01,997):
                    distance=mujoco.mj_ray(actual.model.ptr,actual.data.ptr,np.array([x,.3,5.]),
                        np.array([0.,0.,-1.]),None,1,-1,geomid)
                    if distance<0:raise AssertionError(f'No terrain below x={x}')
                    ray_error=max(ray_error,abs(5-distance-float(task.height_at(x))))
                assert ray_error<5e-6,ray_error
                rows.append(dict(variant=name,terrain_seed=seed,exact_course_arrays=True,
                    model_max_errors=errors,ray_samples=997,max_ray_height_error=ray_error,
                    patches=len(m['mu']),height_samples=len(m['x'])))
            finally:actual.free();expected.free()
    if fixtures:
        destination=Path(fixtures);destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,**arrays)
    result=dict(upstream_commit=UPSTREAM_COMMIT,upstream_blob=blob,
        upstream_sha256=hashlib.sha256(content).hexdigest(),cases=rows,passed=True,
        scope='Course draws and compiled physical model only; reward, sensor, reset, and clock contracts remain benchmark-specific',
        training_updates=0,versions=versions())
    write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-file',required=True,help='environment/cheetah_terrain.py from the pinned upstream commit')
    p.add_argument('--out',default='prechecks/ct_terrain/upstream_match.json')
    p.add_argument('--write-fixtures',help='Optional .npz course fixture output')
    a=p.parse_args();os.environ.setdefault('MUJOCO_GL','egl')
    result=verify(a.upstream_file,a.out,a.write_fixtures)
    print(json.dumps(dict(passed=result['passed'],checked_courses=len(result['cases']),training_updates=0,out=a.out)))


if __name__=='__main__':main()
