"""Finite-course task, termination, operator, and launch-plan checks."""
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from fjbench.config import make_config,observation_dim,select_methods,run_key,calibration_key
from fjbench.path_task import PathTask
from fjbench.data import Replay


def config(case='F',experiment='native',**extra):
    return make_config(case,experiment,overrides=dict(control_task='path_speed',**extra))


class TaskTests(unittest.TestCase):
    def test_target_speed_minimizes_running_cost(self):
        task=PathTask(config(),3.,0.)
        self.assertLess(task.cost_rate(3.,np.zeros(6)),task.cost_rate(6.,np.zeros(6)))
        self.assertLess(task.cost_rate(3.,np.zeros(6)),task.cost_rate(0.,np.zeros(6)))

    def test_metrics_weight_elapsed_time_and_pay_terminal_event_once(self):
        task=PathTask(config(),3.,0.)
        task.record(1.,4.,np.zeros(6),.01,.7,0.)
        task.record(20.,6.,np.zeros(6),.09,.7,0.)
        self.assertAlmostEqual(task.summary(False)['speed_rmse'],math.sqrt(8.2))
        self.assertEqual(task.close_interval(),5.)
        self.assertEqual(task.close_interval(),0.)
        self.assertTrue(task.summary(False)['success'])
        self.assertEqual(task.condition(20.,.1,0.),'fall')
        early=PathTask(config(),3.,0.);early.reason='fall'
        late=PathTask(config(),3.,0.);late.reason='fall';late.elapsed=11.
        self.assertLess(early.close_interval(),late.close_interval())

    def test_goal_measurement_and_task_identity_are_explicit(self):
        c=config();self.assertEqual(observation_dim(c),18)
        self.assertEqual(observation_dim(config(experiment='contact')),32)
        self.assertEqual(c['boundary'],'finite_horizon')
        self.assertAlmostEqual(-math.log(c['gamma_reference'])/c['reference_time'],.05)
        self.assertEqual(config(case='J',experiment='contact')['speed_train'],[3.])
        short=config(path_length=2.)
        self.assertLess(max(short['path_calibration_reset_positions']),2.)

    def test_algorithm_subsets_are_deduplicated_and_preparation_is_shared(self):
        self.assertEqual(select_methods('ct_sac,ph_sac,ct_sac'),['ct_sac','ph_sac'])
        with self.assertRaises(ValueError):select_methods('typo')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            outputs=[]
            for methods in ('ct_sac','ct_sac,ph_sac'):
                path=Path(directory)/('one.jsonl' if methods=='ct_sac' else 'two.jsonl')
                p=subprocess.run([sys.executable,'-S',str(root/'make_fj_plan.py'),'--case','F',
                    '--experiments','contact','--seeds','0','--methods',methods,'--kind','prepare',
                    '--config',str(root/'cases/path_speed.json'),'--out',str(path)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);outputs.append(path.read_text())
            self.assertEqual(outputs[0],outputs[1])

    def test_checkpoint_selection_uses_validation_and_speed_tiebreak(self):
        from select_fj_checkpoint import choose
        from fjbench.checkpoints import checkpoint_path
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'evaluations').mkdir()
            for label,error in [('final',.1),('100',.05)]:
                checkpoint=checkpoint_path(root,label);checkpoint.write_bytes(label.encode())
                for clock in ('fixed','irregular'):
                    value=dict(checkpoint=label,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        config={'seed':0},method='ct_sac',split='validation',cohort='iid',clock=clock,
                        path_metrics={'success_rate':.8,'relative_speed_rmse':error})
                    (root/'evaluations'/f'{label}_validation_iid_{clock}_n2.json').write_text(json.dumps(value))
            (root/'evaluations/final_test_iid_fixed_n2.json').write_text(json.dumps({'path_metrics':{'success_rate':1.}}))
            selected=choose(root,['fixed','irregular'],2)
            self.assertEqual(selected['checkpoint'],'100');self.assertFalse(selected['test_results_used'])
            (root/'selected_checkpoint.json').write_text(json.dumps(selected))
            self.assertEqual(checkpoint_path(root,'selected').name,'policy_000000100.pt')

    def test_aggregation_uses_completion_and_keeps_undefined_success_times(self):
        root_code=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan=[]
            for seed in (0,1):
                c=make_config('F','native',seed,{'control_task':'path_speed'});method='ct_sac'
                key=run_key(c,method);directory=root/key;(directory/'evaluations').mkdir(parents=True)
                checkpoint=directory/'final.pt';checkpoint.write_bytes(b'synthetic fixture')
                plan.append(dict(config=c,method=method,calibration=calibration_key(c),run_id=key))
                value=dict(config=c,method=method,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    family_time_score=9999.,train_counters={},rows=[{'episode_seed':100000,'g':3.}],
                    path_metrics={'success_rate':float(seed),'speed_rmse':3.-seed,'completion_time_success_only':7. if seed else None})
                (directory/'evaluations/final_test_iid_fixed_n1.json').write_text(json.dumps(value))
            path=root/'plan.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in plan));out=root/'out.json'
            p=subprocess.run([sys.executable,str(root_code/'aggregate_fj.py'),'--plan',str(path),'--root',str(root),
                '--clock','fixed','--episodes','1','--out',str(out)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            result=json.loads(out.read_text())['results'][0]
            self.assertEqual(result['primary_metric'],'completion_rate');self.assertEqual(result['mean'],.5)
            self.assertEqual(result['path_metrics']['completion_time_success_only']['mean'],7.)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch operator checks')
class OperatorTests(unittest.TestCase):
    def test_all_path_adapters_handle_requested_holds_and_terminal_boundaries(self):
        import torch
        from fjbench.config import METHODS
        from fjbench.models import InformationMap,OperatorModel,tensors
        from fjbench.agent import Agent
        torch.set_num_threads(1)
        c=config(history=2,encoder_dim=4,nodes=6,action_samples=2,hidden=[16,16],batch=8,fit_steps=1)
        replay=Replay(24,18,2,include_requested_hold=True)
        for i in range(24):
            done=float(i%3==0)
            replay.add(dict(obs=np.zeros(18),next_obs=np.ones(18)*.01,prev_action=np.zeros(6),action=np.zeros(6),
                prev_reward=0.,prev_h=.01,r_src=25.*done,R=5.*done-.001,h=.002 if done else .02,
                h_requested=.03 if done else .02,next_h=.01,terminated=done,truncated=0.,g=3.,time=.1,next_time=.102 if done else .12),i//3)
        information=InformationMap(c,np.zeros(26),np.ones(26)).requires_grad_(False)
        nodes=tensors(replay.windows(range(6)),'cpu');node_info=information.encode_batch(nodes)
        model=OperatorModel(information.feature_dim,{'g':nodes['g'].numpy()},c,[1.,1.])
        batch=tensors(replay.windows(range(8)),'cpu')
        live=1-batch['terminated'];gamma=torch.exp(math.log(c['gamma_reference'])*batch['h']/c['reference_time'])
        raw_gamma=torch.exp(math.log(c['gamma_reference'])*batch['h_requested']/c['reference_time'])
        rows=torch.ones(8,6)/6;rewards=torch.cat([batch['r_src'],batch['R']],1)
        loss,stats=model.loss(information.encode_batch(batch),batch['action'],batch['g'],rows,rewards,
            gamma*live,torch.ones(6,2),0,survival_target=live,raw_gamma=raw_gamma)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(stats['survival_bce'],0)
        for method in METHODS:
            agent=Agent(c,method,information,model,nodes,node_info,'cpu')
            result=agent.update(batch)
            self.assertTrue(math.isfinite(result['critic_loss']))
            if method.startswith('ct_'):self.assertEqual(result['terminal_target_error'],0.)

    def test_ensemble_averages_joint_survival_kernel(self):
        import torch
        from fjbench.models import OperatorModel
        c=config(ensembles=2)
        model=OperatorModel(2,{'g':np.array([[3.],[3.]])},c,[1.,1.])
        with torch.no_grad():
            for i,net in enumerate(model.members):
                for p in net.parameters():p.zero_()
                net[-1].bias[:2]=torch.log(torch.tensor([.9,.1] if i==0 else [.1,.9]))
                net[-1].bias[-1]=math.log(.2/.8) if i==0 else math.log(.8/.2)
        p,_,s=model.predict_with_survival(torch.zeros(1,2),torch.zeros(1,6),torch.tensor([[3.]]))
        np.testing.assert_allclose(s.detach().numpy(),[[.5]],atol=1e-6)
        np.testing.assert_allclose(p.detach().numpy(),[[.26,.74]],atol=1e-6)
        self.assertLessEqual(float((s*p.sum(1,keepdim=True)).max().detach()),1.)

    def test_encoder_and_replay_preserve_requested_hold_at_early_termination(self):
        import torch
        from fjbench.models import InformationMap,tensors
        c=config(history=2,encoder_dim=4)
        replay=Replay(4,18,2,include_requested_hold=True)
        record=dict(obs=np.zeros(18),next_obs=np.ones(18),prev_action=np.zeros(6),action=np.zeros(6),
            prev_reward=0.,prev_h=0.,r_src=1.,R=1.,h=.002,h_requested=.03,next_h=.01,
            terminated=1.,truncated=0.,g=3.,time=0.,next_time=.002)
        replay.add(record,0);recovered=Replay.restore(replay.state())
        information=InformationMap(c,np.zeros(26),np.ones(26))
        feature=information.encode_batch(tensors(recovered.windows([0]),'cpu'))
        self.assertAlmostEqual(float(feature[0,-2].detach()),3.,places=5)


SIM=importlib.util.find_spec('mujoco') and importlib.util.find_spec('dm_control')


@unittest.skipUnless(SIM,'MuJoCo task checks')
class EnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('imageio_ffmpeg'),'Headless evaluator dependencies')
    def test_integrated_path_evaluation_video_preserves_the_same_rollout(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        import torch
        import imageio.v2 as imageio
        from fjbench.models import InformationMap
        from bench.learner import Actor
        from fjbench.evaluation import evaluate
        torch.set_num_threads(1)
        for case in ('F','J'):
            c=config(case,experiment='contact',horizon=.08,episode_step_cap=40,contact_patches=40,hidden=[16,16],encoder_dim=4)
            n=observation_dim(c);information=InformationMap(c,np.zeros(n+8),np.ones(n+8))
            actor=Actor(information.feature_dim,6,c['hidden'])
            packet=dict(config=c,method='ct_sac',calibration_sha256='synthetic-test',counters={},
                public=dict(config=c,information=information.state_dict(),information_mean=np.zeros(n+8),information_scale=np.ones(n+8)),
                agent=dict(actor=actor.state_dict()))
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);torch.save(packet,root/'final.pt')
                baseline=evaluate(c,'ct_sac',root,'final','iid','fixed',1,'cpu')
                recorded=evaluate(c,'ct_sac',root,'final','iid','fixed',1,'cpu',video_options={'episodes':1,'width':320,'height':240,'end_card_seconds':.04})
                self.assertEqual(baseline['path_metrics'],recorded['path_metrics'])
                row=recorded['rows'][0]
                self.assertEqual(row['path_metrics']['terminal_reason'],'timeout')
                self.assertAlmostEqual(row['path_metrics']['initial_x'],0.,places=8)
                self.assertAlmostEqual(row['path_metrics']['trajectory'][-1]['time'],row['elapsed_seconds'])
                with imageio.get_reader(row['video']['video']) as reader:
                    self.assertEqual(reader.count_frames(),3)
                    self.assertEqual(reader.get_data(2).shape,(240,320,3))
                self.assertTrue(any((root/'evaluations').glob('*.png')))

    def test_goal_stops_mid_hold_and_has_zero_continuation(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from fjbench.environment import Environment
        c=config(clock='fixed',fixed_hold=.03)
        env=Environment(c,201,start_x=19.99,initial_speed=3.)
        try:
            _,transition=env.step(np.zeros(6))
            self.assertTrue(transition['terminated']);self.assertFalse(transition['truncated'])
            self.assertLess(transition['h'],.03)
            self.assertEqual(transition['path_event_reward'],5.)
            self.assertTrue(transition['path_metrics']['success'])
            self.assertGreater(transition['R'],4.9)
            with self.assertRaises(RuntimeError):env.step(np.zeros(6))
        finally:env.close()

    def test_timeout_is_finite_and_carries_failure_penalty(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from fjbench.environment import Environment
        env=Environment(config(horizon=.02,episode_step_cap=10),201)
        try:
            for _ in range(10):
                _,transition=env.step(np.zeros(6))
                if transition['terminated'] or transition['truncated']:break
            self.assertEqual(transition['path_metrics']['terminal_reason'],'timeout')
            self.assertEqual(transition['path_event_reward'],-2.)
            self.assertFalse(transition['path_metrics']['success'])
        finally:env.close()


if __name__=='__main__':unittest.main()
