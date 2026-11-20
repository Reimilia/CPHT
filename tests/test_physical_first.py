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
from fjbench.finite import solve,ResidualRecoveryError


class RecoveryTests(unittest.TestCase):
    def test_absolute_gate_can_fail_on_an_accurate_large_value_block(self):
        c=make_config();n,k=8,2
        arrays=(np.zeros((2,n,k)),np.full((n,k),1e5),np.full((n,k),1-1e-4),np.full((n,k,n),1/n),np.zeros((n,k)))
        with self.assertRaises(ResidualRecoveryError):solve(*arrays,dict(c,inverse_relative_tolerance=0.,inverse_refinement_steps=0))
        _,_,report=solve(*arrays,c)
        self.assertTrue(report['inverse_gate_pass']);self.assertFalse(report['inverse_absolute_gate_pass'])
        self.assertLess(report['inverse_relative_residual'],1e-12)
        self.assertGreater(report['recovery_condition_bound'],10000)
        self.assertGreaterEqual(report['linear_solves'],report['phase_steps'])

    def test_nonfinite_rewards_are_rejected_and_numeric_changes_reuse_calibration(self):
        c=make_config();old=dict(c);old.pop('inverse_relative_tolerance');old.pop('inverse_refinement_steps')
        self.assertEqual(calibration_key(c),calibration_key(old))
        with self.assertRaises(FloatingPointError):solve(np.zeros((2,2,1)),np.full((2,1),np.inf),np.full((2,1),.9),np.full((2,1,2),.5),np.zeros((2,1)),c)


TORCH=bool(importlib.util.find_spec('torch'))
SIM=TORCH and bool(importlib.util.find_spec('mujoco')) and bool(importlib.util.find_spec('dm_control'))


@unittest.skipUnless(TORCH,'Torch required')
class UpdateTests(unittest.TestCase):
    def test_a_true_recovery_failure_rejects_the_block_without_changing_networks(self):
        import torch
        from fjbench.agent import Agent
        from fjbench.models import InformationMap,tensors
        from fjbench.data import Replay
        c=make_config('F','native',overrides=dict(control_task='path_speed',hidden=[16,16],encoder_dim=4,history=2))
        n=observation_dim(c);information=InformationMap(c,np.zeros(n+8),np.ones(n+8))
        agent=Agent(c,'ph_sac',information,None,None,None,'cpu')
        replay=Replay(4,n,2,include_requested_hold=True)
        for i in range(4):replay.add(dict(obs=np.zeros(n),next_obs=np.zeros(n),prev_action=np.zeros(6),action=np.zeros(6),
            prev_reward=0.,prev_h=0.,r_src=0.,R=0.,h=.01,h_requested=.01,next_h=.01,terminated=0.,truncated=0.,g=3.,time=0.,next_time=.01),i)
        block=dict(info=torch.zeros((4,information.feature_dim)),actions=torch.zeros((4,6)),initial=torch.zeros((2,2,2)),
            r=torch.zeros((2,2)),gamma=torch.full((2,2),.9),p=torch.full((2,2,2),.5),entropy=torch.zeros((2,2)))
        diagnostic=dict(inverse_residual=1.,inverse_allowed=1e-8,inverse_scale=1.,contraction_gap=.1,operator_calls=2,linear_solves=1)
        before=[p.detach().clone() for p in list(agent.actor.parameters())+list(agent.critics.parameters())]
        with patch.object(agent,'finite_block',return_value=block),patch('fjbench.agent.solve',side_effect=ResidualRecoveryError(diagnostic)):
            stats=agent.update(tensors(replay.windows(range(4)),'cpu'))
        self.assertFalse(stats['critic_admitted']);self.assertEqual(stats['critic_steps'],0);self.assertEqual(stats['actor_steps'],0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,list(agent.actor.parameters())+list(agent.critics.parameters()))))

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
            with patch('fjbench.finite.solve',side_effect=AssertionError('Value inverse called')):report=agent.update(batch)
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

    def test_simple_training_resume_and_cross_environment_evaluation(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        import torch
        from physical_fj.prepare import prepare
        from physical_fj.train import train
        from physical_fj.evaluate import evaluate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,c=self.tiny(root);artifact=root/'calibration';prepare(s,c,artifact)
            with patch('fjbench.finite.solve',side_effect=AssertionError('Value inverse called')):
                train(s,c,'cpht_sac',artifact,root/'reference')
                partial=train(s,c,'cpht_sac',artifact,root/'resumed',pause_after_episodes=1)
                self.assertLess(partial['decisions'],s['total_decisions'])
                train(s,c,'cpht_sac',artifact,root/'resumed',resume=True)
            first=torch.load(root/'reference/final.pt',weights_only=False);second=torch.load(root/'resumed/final.pt',weights_only=False)
            for key,value in first['agent']['actor'].items():torch.testing.assert_close(value,second['agent']['actor'][key],rtol=0,atol=0)
            a=evaluate(root/'reference/final.pt',episodes=1)
            b=evaluate(root/'reference/final.pt',episodes=1,video_episodes=1,width=320,height=240)
            self.assertEqual(a['path_metrics'],b['path_metrics'])
            transferred=evaluate(root/'reference/final.pt',episodes=1,terrain='steps',case_test='J',cohort='layout',clock='irregular')
            self.assertEqual(transferred['config']['case'],'J');self.assertEqual(transferred['config']['terrain_elevation'],'steps')
            self.assertEqual(transferred['calibration_coefficients'],first['calibration']['coefficients'])
            self.assertEqual(transferred['deployment_updates'],0)


if __name__=='__main__':unittest.main()
