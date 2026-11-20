"""Independent mechanics checks and actor-free feedback integration."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

SIM=all(importlib.util.find_spec(n) for n in ('torch','mujoco','dm_control'))


@unittest.skipUnless(SIM,'Torch and MuJoCo required')
class PureMechanicsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from physical_fj.config import settings
        from physical_fj.mechanics import NominalRobot
        from pure_cpht.mechanics import Robot,public_geometry
        torch.set_num_threads(1);self.nominal=NominalRobot(settings())
        self.robot=Robot(public_geometry(self.nominal),dtype=torch.float64)

    def test_mass_bias_energy_and_regression_match_independent_mujoco(self):
        r=self.robot;rng=np.random.default_rng(912)
        for _ in range(5):
            q=rng.uniform(-.4,.4,9);v=rng.normal(size=9);a=rng.normal(size=9)
            actual=self.nominal.evaluate(q,v);d=r.quantities(r.t(q[None]),r.t(v[None]))
            np.testing.assert_allclose(d['M'][0],actual['M0'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(d['c'][0],actual['bias0']+actual['spring0']+actual['damping0'],rtol=1e-12,atol=1e-11)
            self.assertAlmostEqual(float(d['H'][0]),actual['T0']+actual['Vg0']+actual['Vs0'],places=10)
            prediction=r.regressor(r.t(q[None]),r.t(v[None]),r.t(a[None]))@r.parameters
            target=actual['M0']@a+actual['bias0']+actual['spring0']+actual['damping0']
            np.testing.assert_allclose(prediction[0],target,rtol=1e-12,atol=1e-11)
            self.assertGreater(np.linalg.eigvalsh(d['M'][0].numpy()).min(),0.)
        np.testing.assert_array_equal(r.B[:3],0.)

    def test_acceleration_equations_equal_canonical_ph_and_allow_motor_work(self):
        import torch
        r=self.robot;q=r.t([[0.,-.1,.15,.1,-.2,.1,-.3,.1,-.1]]).requires_grad_()
        v=r.t([[.1,0.,.05,.2,-.1,.1,-.1,.2,.1]])
        p=(r.quantities(q,v)['M']@v[:,:,None]).squeeze(-1).detach().requires_grad_()
        d=r.quantities(q,torch.zeros_like(v));H=.5*(p*torch.linalg.solve(d['M'],p[:,:,None]).squeeze(-1)).sum()+d['potential'].sum()
        Hq,Hp=torch.autograd.grad(H,(q,p))
        u=r.t([[.5,-.2,.3,-.1,.4,.1]]);port=r.t([[2.,10.,0.,1.,0.,0.,0.,0.,0.]])
        acceleration=r.acceleration(q.detach(),v,u,port)
        _,Mdotv=torch.autograd.functional.jvp(lambda x:(r.quantities(x,v)['M']@v[:,:,None]).squeeze(-1),q.detach(),v)
        pdot=(r.quantities(q.detach(),v)['M']@acceleration[:,:,None]).squeeze(-1)+Mdotv
        damping=torch.cat((torch.zeros_like(v[:,:3]),r.damping*v[:,3:]),1)
        np.testing.assert_allclose(Hp,v,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(pdot,-Hq-damping+u@r.B.T+port,rtol=1e-10,atol=1e-10)
        derivative=(Hq*v+Hp*pdot).sum();power=(v*(u@r.B.T+port)).sum()-(r.damping*v[:,3:]**2).sum()
        self.assertAlmostEqual(float(derivative),float(power),places=9)
        self.assertGreater(float(power),0.)

    def test_learns_link_perturbations_beyond_the_three_global_scales(self):
        from pure_cpht.learning import fit_mechanics
        from pure_cpht.config import options
        r=self.robot;rng=np.random.default_rng(71);n=96
        q=rng.uniform(-.4,.4,(n,9));v=rng.normal(size=(n,9));a=5*rng.normal(size=(n,9))
        true=r.parameters.numpy()*(1+.15*np.sin(np.arange(32)))
        X=r.regressor(r.t(q),r.t(v),r.t(a)).numpy()
        data=dict(q=q,v=v,acceleration=a,y=X@true,physics_episode=np.repeat(np.arange(12),8))
        fitted,metrics=fit_mechanics(r.geometry,data,r.parameters.numpy(),[4,9],dict(options(),mass_ridge=1e-8))
        self.assertGreater(metrics['design_rank'],20)
        self.assertLess(metrics['validation']['relative_force_rmse'],1e-5)
        self.assertTrue((fitted>0).all())
        self.assertGreater(np.linalg.norm(fitted-r.parameters.numpy()),.01)

    def test_contact_cones_airborne_port_and_causal_force_observer(self):
        import torch
        from pure_cpht.contact import Contact
        from pure_cpht.config import options
        r=self.robot;c=Contact(r,[3000.,100.,.08,.7],options())
        scan=r.t([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.],[3.,0.,0.]])
        q=r.t([[0.,-.1,0.,0.,0.,0.,0.,0.,0.]]);v=torch.zeros_like(q);u=r.t([[.1,0.,0.,0.,0.,0.]])
        d=c.solve(q,v,u,scan)
        self.assertGreater(float(d['normal_force'].sum()),0.)
        self.assertTrue(bool((d['normal_force']>=0).all()))
        self.assertTrue(bool((abs(d['tangent_force'])<=.7*d['normal_force']+1e-10).all()))
        desired=d['port']+.2*(d['J'].transpose(1,2)@d['force'][:,:,None]).squeeze(-1)
        correction,report=c.observe(q,v,u,scan,desired)
        self.assertLess(float(report['observer_error_after'][0]),float(report['observer_error_before'][0]))
        self.assertGreater(float(abs(correction).sum()),0.)
        q[:,1]=2.;air=c.solve(q,v,u,scan,force_correction=correction)
        np.testing.assert_allclose(air['port'],0.,atol=1e-10)

    def test_bounded_feedback_optimizer_matches_independent_scipy(self):
        import torch
        from scipy.optimize import minimize
        from pure_cpht.feedback import bounded_quadratic
        rng=np.random.default_rng(103);matrix=rng.normal(size=(24,6,6))
        Q=matrix.transpose(0,2,1)@matrix+.01*np.eye(6);rhs=10*rng.normal(size=(24,6))
        action,residual,_=bounded_quadratic(torch.tensor(Q),torch.tensor(rhs),32)
        self.assertLess(float(residual.max()),1e-7)
        self.assertLessEqual(float(abs(action).max()),1.)
        for i in range(8):
            reference=minimize(lambda u:.5*u@Q[i]@u-rhs[i]@u,np.zeros(6),jac=lambda u:Q[i]@u-rhs[i],
                bounds=[(-1,1)]*6,method='L-BFGS-B',options=dict(ftol=1e-14,gtol=1e-10))
            value=action[i].numpy();objective=.5*value@Q[i]@value-rhs[i]@value
            self.assertAlmostEqual(objective,reference.fun,places=8)


@unittest.skipUnless(SIM,'Torch and MuJoCo required')
class PurePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from physical_fj.config import settings,control
        from physical_fj.prepare import prepare
        from pure_cpht.learning import prepare_artifact
        os.environ.setdefault('MUJOCO_GL','egl');torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        task=cls.root/'task.json';task.write_text(json.dumps(dict(control_task='path_speed',contact_backend='ct_terrain',
            contact_fixed_seed=1,horizon=.008,episode_step_cap=4)))
        cls.s=settings('F',calibration_decisions=32,total_decisions=40,batch=4,hidden=[16,16],capacity=128,
            save_every=2,log_every=1,pure=dict(horizon_steps=2,substeps=1,candidates=4,elites=2,iterations=1,
                contact_iterations=6,model_refit_every=2))
        cls.c=control(cls.s,task);cls.artifact=cls.root/'artifacts';prepare(cls.s,cls.c,cls.artifact)
        cls.prepared=prepare_artifact(cls.artifact,cls.s)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_probe_contract_baselines_and_feedback_uses_no_future_simulator(self):
        from physical_fj.mechanics import PhysicalEnvironment
        from physical_fj.prepare import load,feature_arrays
        from pure_cpht.controller import Controller
        from physical_fj.train import train
        env=PhysicalEnvironment(self.c,self.s,123,purpose='evaluation')
        try:
            packet=env.current;raw=packet['raw'];scan=packet['scan']
            self.assertEqual(raw.shape,(45,));self.assertEqual(scan.shape,(7,3))
            root=env.base._env.physics.data.xpos[env.base._env.physics.model.name2id('torso','body')]
            measured=np.asarray(env.core.observation[-15:-1]).reshape(2,-1).T
            np.testing.assert_allclose(raw[31:].reshape(7,2),measured,atol=1e-6)
            np.testing.assert_allclose(scan[:,1]-root[2],measured[:,0],atol=1e-6)
            controller=Controller(self.prepared['model'],self.s['pure'],self.prepared['neutral'],device='cpu')
            from pure_cpht.config import options
            unseeded=Controller(self.prepared['model'],options(dict(self.s['pure'],gait_seed_scale=0.)),self.prepared['neutral'])
            np.testing.assert_array_equal(unseeded.mean[6:18],0.)
            with patch('mujoco.mj_forward',side_effect=AssertionError('Future simulator queried')):
                action=controller.act(packet,env.g,env.hold(),0.)
            self.assertTrue(np.isfinite(action).all());self.assertLessEqual(abs(action).max(),1.)
            controller.first_hold=.03;before_steps=controller.predicted_steps
            with patch.object(controller,'feedback',wraps=controller.feedback) as calls:
                controller.rollout(controller.robot.t(raw[:9])[None],controller.robot.t(raw[9:18])[None],
                    controller.robot.t(scan),controller.mean[None].expand(4,-1),env.g,0.)
            self.assertEqual(calls.call_count,2)
            self.assertEqual(controller.predicted_steps-before_steps,4*(3+1))
        finally:env.close()
        report,data,norm=load(self.artifact);features=feature_arrays(data,report['coefficients'])
        self.assertEqual(features['ph'].shape[1],49)
        result=train(self.s,self.c,'ct_sac',self.artifact,self.root/'ct_with_matching_probes')
        self.assertEqual(result['linear_value_solves'],0)

    def test_actor_free_learning_resume_and_recording_invariance(self):
        import torch
        from pure_cpht.training import train
        from physical_fj.evaluate import evaluate
        with patch('physical_fj.learner.Learner',side_effect=AssertionError('SAC learner constructed')):
            first=train(self.s,self.c,'pure_cpht',self.artifact,self.root/'full')
            partial=train(self.s,self.c,'pure_cpht',self.artifact,self.root/'resumed',pause_after_episodes=1)
            self.assertLess(partial['decisions'],self.s['total_decisions'])
            second=train(self.s,self.c,'pure_cpht',self.artifact,self.root/'resumed',resume=True)
        self.assertEqual(first['actor_updates'],0);self.assertEqual(first['critic_updates'],0)
        self.assertGreater(first['model_updates'],0);self.assertGreater(first['control_optimizations'],0)
        a=torch.load(self.root/'full/final.pt',weights_only=False);b=torch.load(self.root/'resumed/final.pt',weights_only=False)
        self.assertIsNone(a['actor']);self.assertIsNone(a['critic'])
        np.testing.assert_array_equal(a['pure_model']['parameters'],b['pure_model']['parameters'])
        np.testing.assert_array_equal(a['gait'],b['gait']);self.assertEqual(first,second)
        x=evaluate(self.root/'full/final.pt',episodes=1)
        y=evaluate(self.root/'full/final.pt',episodes=1,video_episodes=1,width=320,height=240)
        self.assertEqual(x['path_metrics'],y['path_metrics'])
        self.assertEqual(x['rows'][0]['prediction_metrics'],y['rows'][0]['prediction_metrics'])
        self.assertEqual(y['deployment_updates'],0);self.assertFalse(y['actor'])
        z=evaluate(self.root/'full/final.pt',episodes=1,case_test='J',terrain='steps',cohort='layout',clock='irregular')
        self.assertEqual(z['config']['case'],'J');self.assertEqual(z['config']['terrain_elevation'],'steps')
        self.assertTrue(z['model_frozen']);self.assertEqual(z['rows'][0]['model_updates'],0)
        feedback=evaluate(self.root/'full/final.pt',episodes=1,feedback_only=True)
        self.assertEqual(feedback['method'],'pure_cpht_feedback')
        self.assertEqual(feedback['rows'][0]['controller']['control_optimizations'],0)
        self.assertTrue(feedback['model_frozen']);self.assertEqual(feedback['checkpoint_sha256'],x['checkpoint_sha256'])


if __name__=='__main__':unittest.main()
