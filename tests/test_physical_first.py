"""Finite precision recovery and the separate instrumented physical-first pilot."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fjbench.config import make_config,calibration_key,observation_dim




TORCH=bool(importlib.util.find_spec('torch'))
SIM=TORCH and bool(importlib.util.find_spec('mujoco')) and bool(importlib.util.find_spec('dm_control'))


@unittest.skipUnless(TORCH,'Torch required')
class UpdateTests(unittest.TestCase):

    def test_all_simple_controls_use_measured_targets_without_a_value_inverse(self):
        import torch
        from physical_fj.config import settings,control
        from physical_fj.learner import Learner
        torch.set_num_threads(1)
        s=settings('F',hidden=[16,16],batch=4);c=control(s)
        for method in ('ct_sac','cpht_sac','sac','ct_sac_ph'):
            dimension=35 if method in ('cpht_sac','ct_sac_ph') else 31
            agent=Learner(dimension,s,c,method,'cpu')
            batch=dict(obs=torch.randn(4,dimension),next_obs=torch.randn(4,dimension),action=torch.zeros(4,6),
                R=torch.full((4,1),5.),r_src=torch.full((4,1),25.),h=torch.full((4,1),.002),
                h_requested=torch.full((4,1),.01),next_h=torch.full((4,1),.01),done=torch.ones(4,1))
            report=agent.update(batch)  # this learner has no value-inverse solver
            self.assertEqual(report['terminal_target_error'],0.);self.assertEqual(report['linear_value_solves'],0)


@unittest.skipUnless(SIM,'Torch and MuJoCo required')
class PhysicalTests(unittest.TestCase):
    def tiny(self,root,case='F',total=40):
        from physical_fj.config import settings,control
        options=root/'task.json';options.write_text(json.dumps(dict(control_task='path_speed',contact_backend='ct_terrain',contact_fixed_seed=1,horizon=.008,episode_step_cap=4)))
        s=settings(case,calibration_decisions=32,total_decisions=total,batch=4,hidden=[16,16],capacity=128,log_every=1,save_every=4)
        return s,control(s,options)

    def test_failed_calibration_preserves_and_reuses_physics_acquisition(self):
        from physical_fj.prepare import prepare,load
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,c=self.tiny(root);artifact=root/'calibration'
            with patch('physical_fj.prepare.fit',side_effect=RuntimeError('planned fit interruption')):
                with self.assertRaisesRegex(RuntimeError,'planned fit interruption'):prepare(s,c,artifact)
            self.assertTrue((artifact/'data.npz').exists());self.assertFalse((artifact/'calibration.json').exists())
            original=(artifact/'data.npz').read_bytes()
            with patch('physical_fj.prepare.acquire',side_effect=AssertionError('Acquisition repeated')):
                report=prepare(s,c,artifact)
            self.assertEqual(original,(artifact/'data.npz').read_bytes())
            self.assertFalse((artifact/'calibration_failure.json').exists())
            self.assertEqual(report['counters']['decisions'],32)
            _,data,norm=load(artifact)
            train_ids=set(data['episode'][norm['train_indices']]);val_ids=set(data['episode'][norm['validation_indices']])
            self.assertFalse(train_ids&val_ids)
            self.assertEqual(set(data['physics_episode']),train_ids|val_ids)

    def test_three_physical_coefficients_and_point_jacobian_coupling(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from physical_fj.config import settings,control
        from physical_fj.prepare import prepare,load
        from physical_fj.mechanics import NominalRobot
        for case in ('F','J'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);s=settings(case,calibration_decisions=64,total_decisions=80,batch=8,hidden=[16,16]);c=control(s)
                report=prepare(s,c,root);coeff=report['coefficients']
                np.testing.assert_allclose(coeff,[1/.8,1/1.2,1/1.4],rtol=1e-8,atol=1e-8)
                self.assertLess(report['metrics']['validation']['force_rmse'],1e-7)
                self.assertGreater(report['max_off_diagonal_mass_norm'],.1)
                _,data,norm=load(root);self.assertEqual(data['raw'].shape[1],31)
                self.assertEqual(len(data['X']),report['counters']['physics_steps'])
                self.assertGreater(len(data['X']),len(data['raw']))
                self.assertEqual(len(norm['ph_mean']),35)
                self.assertFalse(set(data['episode'][norm['train_indices']])&set(data['episode'][norm['validation_indices']]))
                examples=json.loads((root/'contact_examples.json').read_text());coupled=False
                for row in examples:
                    for contact in row['contacts']:
                        direct=np.asarray(contact['generalized_force']);response=np.asarray(contact['calibrated_acceleration_response'])
                        if contact['body']=='bfoot':
                            np.testing.assert_allclose(direct[6:9],0,atol=1e-10);coupled|=np.linalg.norm(response[6:9])>1e-6
                self.assertTrue(coupled)
                # Check the calibrated physical energy derivative independently
                # of the regression's force-balance residual.
                i=len(data['X'])-1;q=data['q'][i];v=data['v'][i]
                nominal=NominalRobot(s);basis=nominal.evaluate(q,v)
                acceleration=data['acceleration'][i]
                def energy(position):
                    b=nominal.evaluate(position,v)
                    return coeff[0]*(b['T0']+b['Vg0'])+coeff[1]*b['Vs0']
                gradient=np.array([(energy(q+1e-5*np.eye(9)[j])-energy(q-1e-5*np.eye(9)[j]))/(2e-5) for j in range(9)])
                derivative=float(gradient@v+coeff[0]*v@basis['M0']@acceleration)
                port=float(v@data['y'][i]-coeff[2]*basis['dissipation0'])
                self.assertAlmostEqual(derivative,port,delta=1e-3*(1+abs(port)))

    def test_force_sensor_queries_leave_the_simulated_trajectory_unchanged(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from physical_fj.config import settings,control
        from physical_fj.mechanics import PhysicalEnvironment
        from fjbench.environment import Environment
        s=settings('F',calibration_decisions=64,total_decisions=80,batch=8);c=control(s)
        plain=Environment(c,123,purpose='evaluation');instrumented=PhysicalEnvironment(c,s,123,purpose='evaluation')
        try:
            for i in range(4):
                action=np.full(6,.1*np.sin(i));_,info=plain.step(action);_,other,_=instrumented.step(action,calibrate=True)
                for key in ('qpos','qvel','qacc_warmstart'):
                    np.testing.assert_array_equal(getattr(plain.base._env.physics.data,key),getattr(instrumented.base._env.physics.data,key))
                self.assertEqual(info['R'],other['R'])
        finally:plain.close();instrumented.close()



if __name__=='__main__':unittest.main()
