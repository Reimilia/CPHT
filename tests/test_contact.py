"""Map interventions, causal sensor boundaries, and optional simulator checks."""
import importlib.util
import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from fjbench.config import make_config,observation_dim,calibration_key,run_key
from bench.protocol import ROOT
from fjbench.contact import sample_map,query,measure,build_xml,curriculum_level


class MapTests(unittest.TestCase):
    def test_F_map_and_materials_stay_fixed_across_resets(self):
        c=make_config("F","contact")
        self.assertEqual(sample_map(c,1),sample_map(c,9031,progress=0))
        with self.assertRaises(ValueError):sample_map(c,1,"terrain")

    def test_J_holdouts_change_one_physical_factor(self):
        c=make_config("J","contact");base=sample_map(c,311)
        terrain=sample_map(c,311,"terrain");friction=sample_map(c,311,"friction")
        material=sample_map(c,311,"material")
        for other in (terrain,friction,material):
            self.assertEqual(base["map_seed"],other["map_seed"])
        for name in ("mu","tau","zeta"):np.testing.assert_equal(base[name],terrain[name])
        for name in ("x","z","tau","zeta"):np.testing.assert_equal(base[name],friction[name])
        for name in ("x","z","mu"):np.testing.assert_equal(base[name],material[name])
        self.assertFalse(np.allclose(base["z"],terrain["z"]))
        self.assertGreater(min(base["mu"]),max(friction["mu"]))
        self.assertLess(max(base["tau"]),min(material["tau"]))

    def test_layout_and_validation_banks_are_disjoint(self):
        c=make_config("J","contact")
        banks={cohort:{sample_map(c,s,cohort)["map_seed"] for s in range(40)}
               for cohort in ("iid","validation","layout")}
        self.assertFalse(banks["iid"]&banks["validation"])
        self.assertFalse(banks["iid"]&banks["layout"])
        self.assertFalse(banks["layout"]&banks["validation"])

    def test_sensors_use_relative_geometry_and_explicit_material_mode(self):
        c=make_config("J","contact",overrides={"contact_height_std":0.,"contact_slope_std":0.})
        m=sample_map(c,7);r=np.random.default_rng(1)
        y=measure(c,m,4.,.7,r)
        self.assertEqual(len(y),14);self.assertEqual(observation_dim(c),31)
        np.testing.assert_allclose(y[:7],query(m,4+np.asarray(c["contact_offsets"]))["height"]-.7)
        # Replacing hidden contact parameters leaves geometry-only data intact.
        changed=dict(m,mu=[.13]*len(m["mu"]),tau=[.08]*len(m["mu"]))
        np.testing.assert_array_equal(y,measure(c,changed,4.,.7,r))
        history=make_config("J","contact",overrides={"contact_sensor":"history"})
        self.assertEqual(observation_dim(history),17);self.assertEqual(len(measure(history,m,0,0,r)),0)
        material=make_config("J","contact",overrides={"contact_sensor":"material"})
        self.assertEqual(observation_dim(material),52)
        self.assertNotEqual(calibration_key(c),calibration_key(material))

    def test_profiles_separate_geometry_friction_and_compliance(self):
        for profile in ("elevation","friction","material"):
            c=make_config("J","contact",overrides={"contact_profile":profile})
            m=sample_map(c,19)
            if profile!="elevation":np.testing.assert_allclose(m["z"],0.)
            if profile in ("elevation","friction"):
                np.testing.assert_allclose(m["tau"],.02);np.testing.assert_allclose(m["zeta"],1.)

    def test_continuous_wedge_edges_and_collision_precedence(self):
        c=make_config("J","contact");m=sample_map(c,61)
        source='<mujoco><worldbody><geom name="floor" type="plane"/></worldbody></mujoco>'
        root=ET.fromstring(build_xml(source,m))
        self.assertEqual(root.find("worldbody/geom").get("contype"),"0")
        geometries=root.findall("worldbody/geom")[1:]
        self.assertTrue(all(g.get("priority")=="2" and g.get("condim")=="3" for g in geometries))
        meshes=root.findall("asset/mesh")
        first=np.fromstring(meshes[5].get("vertex"),sep=" ").reshape(8,3)
        second=np.fromstring(meshes[6].get("vertex"),sep=" ").reshape(8,3)
        np.testing.assert_allclose(first[[6,7]],second[[4,5]])
        top=np.cross(first[6]-first[4],first[7]-first[4])
        self.assertGreater(top[2],0)
        faces=np.fromstring(meshes[5].get("face"),sep=" ",dtype=int).reshape(-1,3)
        volume=sum(np.dot(first[a],np.cross(first[b],first[c])) for a,b,c in faces)/6
        self.assertGreater(volume,0)

    def test_curriculum_is_fixed_for_F_and_full_at_evaluation(self):
        f=make_config("F","contact");j=make_config("J","contact")
        self.assertEqual(curriculum_level(f,0),1.)
        self.assertEqual(curriculum_level(j,0),.35)
        self.assertEqual(curriculum_level(j,.2),.65)
        self.assertEqual(curriculum_level(j,1),1.)
        self.assertEqual(sample_map(j,2,"validation",progress=0)["difficulty"],1.)
        with self.assertRaises(ValueError):make_config("F","contact",overrides={"contact_curriculum":True})


class AggregationTests(unittest.TestCase):
    def test_contact_metrics_average_tasks_before_training_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan=[]
            for seed in (0,1):
                c=make_config("J","contact",seed);method="interval_euler"
                key=run_key(c,method);directory=root/key
                (directory/"evaluations").mkdir(parents=True)
                checkpoint=directory/"final.pt";checkpoint.write_bytes(b"synthetic unit-test fixture")
                plan.append(dict(config=c,method=method,calibration=calibration_key(c),run_id=key))
                rows=[dict(episode_seed=100000+i,g=g,time_score=1.,
                           contact_audit={"low_torso_fraction":float(g==10)})
                      for i,g in enumerate((6,10,10,14))]
                value=dict(config=c,method=method,rows=rows,family_time_score=1.,train_counters={},
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                (directory/"evaluations/final_test_iid_irregular_n4.json").write_text(json.dumps(value))
            path=root/"plan.jsonl";path.write_text("".join(json.dumps(x)+"\n" for x in plan))
            out=root/"report.json"
            process=subprocess.run([sys.executable,str(ROOT/"aggregate_fj.py"),"--plan",str(path),
                "--root",str(root),"--episodes","4","--out",str(out)],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            metrics=json.loads(out.read_text())["results"][0]["contact_metrics"]
            self.assertAlmostEqual(metrics["low_torso_fraction"]["mean"],1/3)
            self.assertEqual(set(metrics["low_torso_fraction"]["per_training_seed"]),{"0","1"})


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("dm_control"),
                     "Optional MuJoCo contact checks")
class ContactBackendTests(unittest.TestCase):
    def test_low_ground_coefficient_and_foot_Jacobian_support(self):
        os.environ.setdefault("MUJOCO_GL","egl")
        from fjbench.environment import Environment
        c=make_config("F","contact",overrides={"contact_mu_train":[.2,.2],"contact_patches":40})
        env=Environment(c,201,start_x=0.)
        try:
            for _ in range(5):obs,info=env.step(np.zeros(6))
            audit=info["contact_audit"]
            self.assertEqual(obs.shape,(31,))
            self.assertLess(audit["max_effective_friction_error"],1e-10)
            self.assertLess(audit["max_time_constant_error"],1e-10)
            self.assertLess(audit["max_damping_ratio_error"],1e-10)
            feet=[r for r in audit["last_contacts"] if r["body"] in ("bfoot","ffoot")]
            self.assertEqual({r["body"] for r in feet},{"bfoot","ffoot"})
            for r in feet:
                self.assertAlmostEqual(r["mu_effective"],.2)
                tau=np.array(r["generalized_force"])
                np.testing.assert_allclose(tau[6:9] if r["body"]=="bfoot" else tau[3:6],0,atol=1e-10)
            self.assertTrue(np.isfinite(obs).all())
            data=env.base._env.physics.data
            before=[getattr(data,k).copy() for k in ("qpos","qvel","qacc_warmstart","qfrc_constraint")]
            env.contact_audit.sample(c["physics_dt"])
            for key,expected in zip(("qpos","qvel","qacc_warmstart","qfrc_constraint"),before):
                np.testing.assert_array_equal(getattr(data,key),expected)
        finally:env.close()

    def test_random_map_reset_and_extra_sensor_do_not_change_actuators(self):
        os.environ.setdefault("MUJOCO_GL","egl")
        from fjbench.environment import Environment
        for mode in ("history","material"):
            c=make_config("J","contact",overrides={"contact_sensor":mode,"contact_patches":40})
            env=Environment(c,503,start_x=5.)
            try:
                obs,info=env.step(np.zeros(6))
                self.assertEqual(obs.shape,(observation_dim(c),))
                self.assertEqual(env.base._env.physics.model.nu,6)
                self.assertGreater(info["h"],0)
            finally:env.close()


if __name__=="__main__":unittest.main()
