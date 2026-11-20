"""Independent course fixtures and physical checks for the CT terrain adapter."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import numpy as np
from bench.protocol import ROOT
from fjbench.config import config_from,make_config,observation_dim,calibration_key
from fjbench.contact import sample_map,query,validate_map


def config(case='F',name='terrain',**extra):
    return config_from(ROOT/f'tests/fixtures/cases/path_speed_ct_{name}.json',case,'contact',extra=extra)


class CourseTests(unittest.TestCase):
    def test_all_course_arrays_match_independent_pinned_source_fixtures(self):
        # Created by verify_terrain_match.py from the hash-checked upstream
        # module. These arrays are independent of this adapter's generator.
        with np.load(ROOT/'tests/fixtures/ct_terrain_reference.npz') as reference:
            for name in ('hills','friction','terrain','slopes','steps'):
                for seed in (1,7):
                    m=sample_map(config('J',name),seed)
                    for field in ('x','z','mu','patch_edges'):
                        np.testing.assert_array_equal(m[field],reference[f'{name}_{seed}_{field}'])
                    self.assertEqual(len(m['mu']),52);self.assertEqual(m['ncol'],81)
                    self.assertEqual((m['x'][0],m['x'][-1]),(-5.,203.))
                    self.assertEqual(m['priority'],1)

    def test_F_fixed_course_and_J_fresh_paired_courses(self):
        f=config();a=sample_map(f,4);b=sample_map(f,999,'validation')
        self.assertEqual(a['map_seed'],1);self.assertEqual(a['map_sha256'],b['map_sha256'])
        j=config('J');first=sample_map(j,17)
        self.assertEqual(first,sample_map(j,17,progress=0.))
        self.assertNotEqual(first['z'],sample_map(j,18)['z'])
        self.assertNotEqual(first['map_seed'],sample_map(j,17,'layout')['map_seed'])
        self.assertFalse(j['contact_curriculum']);self.assertEqual(j['contact_calibration_levels'],[1.])
        # Legacy F validation also keeps the fixed physical map.
        legacy=make_config('F','contact')
        self.assertEqual(sample_map(legacy,1)['map_sha256'],sample_map(legacy,2,'validation')['map_sha256'])

    def test_source_start_pad_and_random_number_order_are_preserved(self):
        hills=sample_map(config(name='hills'),4);combined=sample_map(config(),4)
        flat=sample_map(config(name='friction'),4)
        np.testing.assert_array_equal(hills['z'],combined['z'])
        self.assertFalse(np.array_equal(flat['mu'],combined['mu']))
        for m in (hills,combined,flat):
            np.testing.assert_allclose(np.asarray(m['z'])[np.asarray(m['x'])<=3],0)
            np.testing.assert_allclose(np.asarray(m['mu'])[np.asarray(m['patch_edges'])[:-1]<3],.4)
            self.assertTrue(np.all((np.asarray(m['mu'])>=.1)&(np.asarray(m['mu'])<=1.)))
        np.testing.assert_allclose(hills['mu'],.4)
        self.assertAlmostEqual(max(abs(np.asarray(hills['z']))),.3)
        np.testing.assert_allclose(flat['z'],0)

    def test_no_hidden_material_curriculum_or_observation_change(self):
        c=config();self.assertEqual(observation_dim(c),32)
        legacy=make_config('F','contact',overrides={'control_task':'path_speed'})
        self.assertEqual(observation_dim(c),observation_dim(legacy))
        self.assertEqual(c['contact_offsets'],legacy['contact_offsets'])
        self.assertNotEqual(calibration_key(c),calibration_key(legacy))
        for extra in ({'contact_curriculum':True},{'contact_profile':'material'},{'contact_robot_variation':True},{'terrain_reference_commit':'changed'}):
            with self.assertRaises(ValueError):config(**extra)
        with self.assertRaises(ValueError):sample_map(config('J'),1,'material')
        with self.assertRaises(ValueError):sample_map(config('J'),1,difficulty=.5)
        m=sample_map(c,4);m['z'][20]=.2
        with self.assertRaises(ValueError):validate_map(m)



SIM=importlib.util.find_spec('mujoco') and importlib.util.find_spec('dm_control')


@unittest.skipUnless(SIM,'MuJoCo/dm-control required')
class PhysicalTests(unittest.TestCase):
    def test_surface_rays_match_the_prescribed_profiles(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        import mujoco
        from fjbench.environment import Environment
        for name in ('hills','friction','terrain','slopes','steps'):
            env=Environment(config(name=name),0)
            try:
                p=env.base._env.physics;m=env.contact_map;gid=np.zeros(1,np.int32)
                for x in np.linspace(-4.99,202.99,257):
                    distance=mujoco.mj_ray(p.model.ptr,p.data.ptr,np.array([x,.3,5.]),np.array([0.,0.,-1.]),None,1,-1,gid)
                    self.assertGreaterEqual(distance,0.)
                    self.assertAlmostEqual(5-distance,float(query(m,[x])['height'][0]),places=5)
                self.assertEqual(p.model.nhfield,52);self.assertEqual((p.model.nq,p.model.nv,p.model.nu),(9,9,6))
                self.assertAlmostEqual(float(p.model.body_mass.sum()),14.,places=8)
                self.assertEqual(env.backend['ground_priority'],1)
            finally:env.close()

    def test_icy_patch_uses_its_own_coefficient_and_stock_contact_response(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from fjbench.environment import Environment
        c=config(name='friction');m=sample_map(c,0);edges=np.asarray(m['patch_edges']);mu=np.asarray(m['mu'])
        candidates=np.flatnonzero((edges[:-1]>=3)&(edges[1:]<20))
        icy=int(candidates[np.argmin(mu[candidates])]);self.assertLess(mu[icy],.4)
        env=Environment(c,21,start_x=float((edges[icy]+edges[icy+1])/2))
        try:
            contacts=[]
            for _ in range(5):
                _,info=env.step(np.zeros(6));contacts.extend(info['contact_audit']['last_contacts'])
            self.assertTrue(contacts)
            for row in contacts:
                self.assertAlmostEqual(row['mu_effective'],float(mu[icy]),places=7)
                self.assertEqual(row['solref'],[.02,1.])
        finally:env.close()




if __name__=='__main__':unittest.main()
