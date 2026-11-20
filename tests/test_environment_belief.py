"""Sensor isolation, articulated Jacobians, causal estimation, and checkpoint use."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from environment_belief.config import options, contract

SIM = all(importlib.util.find_spec(n) for n in ('torch', 'mujoco', 'dm_control', 'gymnasium'))


class ContractTests(unittest.TestCase):
    def test_legacy_identity_and_explicit_new_modes(self):
        from physical_fj.config import settings, control, calibration_id
        old = settings('F')
        original = calibration_id(old, control(old))
        self.assertNotIn('sensing', old)
        ids = set()
        for mode, dimension in (('state', 31), ('state_belief', 60), ('sensed_belief', 60)):
            s = settings('F', pure={}, sensing=dict(mode=mode))
            self.assertEqual(contract(s)['observation_dimension'], dimension)
            ids.add(calibration_id(s, control(s)))
        self.assertEqual(len(ids), 3)
        self.assertNotIn(original, ids)
        with self.assertRaises(ValueError):options(dict(sensor_dropout=1.1))
        with self.assertRaises(ValueError):options(dict(friction_events=[dict(time=2., scale=.5), dict(time=1., scale=1.)]))


@unittest.skipUnless(SIM, 'Torch/MuJoCo runtime required')
class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from physical_fj.config import settings
        from physical_fj.mechanics import NominalRobot
        from pure_cpht.mechanics import public_geometry
        os.environ.setdefault('MUJOCO_GL', 'egl')
        cls.nominal = NominalRobot(settings())
        cls.geometry = public_geometry(cls.nominal)
        cls.offsets = [-1., -.5, 0., .5, 1., 2., 3.]

    def observer(self, **extra):
        from environment_belief.observer import Observer
        return Observer(self.geometry, self.offsets, options(extra))

    def test_quadratic_recovery_curvature_transport_and_age(self):
        observer = self.observer(height_std=.001, slope_std=.001)
        observer.predict(0., 0.)
        x = np.asarray(self.offsets)
        a = np.array([.2, .12, .07])
        observer.assimilate_geometry(dict(time=0., x=x, height=a[0]+a[1]*x+.5*a[2]*x*x, slope=a[1]+a[2]*x))
        np.testing.assert_allclose(observer.a, a, atol=1e-4)
        observer.predict(.04, .3)
        scan, features, info = observer.packet(.6)
        self.assertEqual(features.shape, (29,))
        self.assertAlmostEqual(info['height'], .2+.12*.3+.5*.07*.3**2, places=4)
        self.assertAlmostEqual(info['curvature'], .07/(1+(.12+.07*.3)**2)**1.5, places=4)
        self.assertAlmostEqual(info['geometry_sensor_age'], .04)
        self.assertTrue(info['curvature_valid'])
        self.assertGreater(np.linalg.eigvalsh(observer.P).min(), 0.)
        with self.assertRaises(ValueError):observer.predict(.01, .3)

    def test_friction_requires_loaded_dissipative_sliding_and_retains_uncertainty(self):
        observer = self.observer()
        observer.predict(0., 0.)
        prior = observer.log_mu
        for time in (0., .04, .08):
            observer.predict(time, 0.)
            observer.friction_update(100., -20., 0., .01, time)
        self.assertEqual(observer.log_mu, prior)
        self.assertEqual(observer.friction_updates, 0)
        for i in range(1, 9):
            time = .08+.03*i
            observer.predict(time, 0.)
            observer.friction_update(100., -25., .5, .01, time)
        self.assertAlmostEqual(np.exp(observer.log_mu), .25, delta=.025)
        self.assertGreater(observer.friction_updates, 0)
        before = observer.log_mu
        observer.friction_update(0., 0., 1., 0., .4)
        observer.friction_update(100., -25., .5, .9, .4)
        self.assertEqual(observer.log_mu, before)
        observer.predict(.5, 2.)
        self.assertIsNone(observer.friction_time)
        self.assertGreaterEqual(observer.mu_variance, options({})['friction_log_std']**2)

    def test_numpy_point_jacobian_matches_independent_mujoco(self):
        import mujoco
        from environment_belief.observer import point_kinematics
        rng = np.random.default_rng(532)
        m, d = self.nominal.model, self.nominal.data
        for _ in range(3):
            q = rng.uniform(-.3, .3, 9)
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            p, J, _, _, _ = point_kinematics(self.geometry, q)
            for i, item in enumerate(self.geometry['points']):
                body = item['body']
                local = np.array([item['local'][0], 0., item['local'][1]])
                world = d.xpos[body]+d.xmat[body].reshape(3, 3)@local
                actual = np.zeros((3, 9))
                mujoco.mj_jac(m, d, actual, np.zeros_like(actual), world, body)
                np.testing.assert_allclose(p[i], world[[0, 2]], atol=1e-12)
                np.testing.assert_allclose(J[i], actual[[0, 2]], atol=1e-12)


@unittest.skipUnless(SIM, 'Torch/MuJoCo runtime required')
class SensingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        os.environ.setdefault('MUJOCO_GL', 'egl')
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.task = cls.root/'task.json'
        cls.task.write_text(json.dumps(dict(control_task='path_speed', contact_backend='ct_terrain',
            contact_fixed_seed=1, horizon=.08, episode_step_cap=40)))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def settings(self, mode):
        from physical_fj.config import settings, control
        extra = json.loads(Path('cases/sensing_smoke.json').read_text())
        extra['sensing']['mode'] = mode
        s = settings('F', **extra)
        return s, control(s, self.task)

    def test_original_state_does_not_read_probe_buffer_or_private_map(self):
        from physical_fj.mechanics import PhysicalEnvironment
        for mode in ('state', 'state_belief'):
            s, c = self.settings(mode)
            env = PhysicalEnvironment(c, s, 921)
            try:
                env.core.observation = object()
                with patch('fjbench.contact.query', side_effect=AssertionError('Private map read by a state-only observer')):
                    packet = env.observe()
                self.assertEqual(len(packet['raw']), 31 if mode=='state' else 60)
                self.assertFalse(packet['environment']['geometry_sensor_valid'])
                self.assertEqual(env.sensing.frames_attempted, 0)
                self.assertEqual(packet['scan_source'], 'causal_environment_observer')
            finally:env.close()

    def test_sensor_latency_physical_clock_hidden_switch_and_episode_pairing(self):
        from physical_fj.mechanics import PhysicalEnvironment
        s, c = self.settings('sensed_belief')
        env = PhysicalEnvironment(c, s, 918, clock='fixed')
        try:
            first = env.current
            self.assertFalse(first['environment']['geometry_sensor_valid'])
            initial = env.core.base._env.physics.model.geom_friction[env.sensing.ground].copy()
            for _ in range(3):env.step(np.zeros(6))
            np.testing.assert_allclose(env.core.base._env.physics.model.geom_friction[env.sensing.ground], initial*.4)
            self.assertTrue(env.current['environment']['geometry_sensor_valid'])
            self.assertGreaterEqual(env.current['environment']['geometry_sensor_age'], s['sensing']['sensor_delay']-1e-9)
            self.assertEqual(env.sensing.frames_attempted, int(np.floor(env.base.cur_t/.004+1e-7))+1)
            self.assertEqual(env.sensing.events_applied[0]['applied_time'], .02)
            self.assertNotIn('friction_events', env.current['information_contract'])
        finally:env.close()

    def test_all_modes_train_both_methods_and_deploy_frozen_checkpoints(self):
        from physical_fj.prepare import prepare
        from physical_fj.train import train as train_sac
        from pure_cpht.training import train as train_pure
        from physical_fj.evaluate import evaluate
        from compare_sensing_fj import compare
        reports = []
        for mode in ('state', 'state_belief', 'sensed_belief'):
            s, c = self.settings(mode)
            artifact = self.root/mode/'artifact'
            calibration = prepare(s, c, artifact)
            with np.load(artifact/'data.npz') as data:
                self.assertNotIn('contact_sample', data.files)
            expected = 31 if mode=='state' else 60
            self.assertEqual(calibration['raw_observation_dimension'], expected)
            for method, trainer in (('ct_sac', train_sac), ('pure_cpht', train_pure)):
                run = self.root/mode/method
                counters = trainer(s, c, method, artifact, run)
                self.assertEqual(counters['policy_decisions'], 4)
                report = evaluate(run/'final.pt', episodes=1, video_episodes=0)
                self.assertEqual(report['information_contract']['observation_dimension'], expected)
                self.assertEqual(report['observer_parameter_updates'], 0)
                self.assertEqual(report['deployment_updates'], 0)
                self.assertEqual(report['rows'][0]['assimilation_metrics']['sensor']['attempted']>0, mode=='sensed_belief')
                reports.append(report)
        import torch
        state=torch.load(self.root/'state/pure_cpht/final.pt',weights_only=False)
        shared=torch.load(self.root/'state_belief/pure_cpht/final.pt',weights_only=False)
        np.testing.assert_array_equal(state['gait'],shared['gait'])
        np.testing.assert_array_equal(state['pure_model']['contact'],shared['pure_model']['contact'])
        self.assertFalse(state['model_metrics']['contact']['point_contact_labels_used'])
        result = compare(reports)
        self.assertEqual(len(result['results']), 6)
        self.assertEqual(len(result['paired_differences']), 7)
        bad = json.loads(json.dumps(reports))
        bad[0]['evaluation_sensing']['sensor_delay'] = .123
        with self.assertRaises(ValueError):compare(bad)
        checkpoint = self.root/'state/ct_sac/final.pt'
        with self.assertRaises(ValueError):evaluate(checkpoint, episodes=1, sensing_test=dict(mode='sensed_belief'))
        for method in ('ct_sac','pure_cpht'):
            report=evaluate(self.root/'sensed_belief'/method/'final.pt',episodes=1,clock='irregular',case_test='J',
                sensing_test=dict(sensor_delay=.012,sensor_dropout=.2))
            self.assertEqual(report['config']['case'],'J')
            self.assertEqual(report['clock'],'irregular')
            self.assertEqual(report['observer_parameter_updates'],0)


if __name__=='__main__':unittest.main()
