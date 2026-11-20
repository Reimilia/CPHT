"""Sparse goal coverage, disjoint splits, saved acquisition, and bounded probes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from bench.protocol import ROOT
from fjbench.data import Replay
from fjbench.config import make_config,config_from,observation_dim


def csmall(case='F',backend='ct_terrain',**extra):
    values=dict(control_task='path_speed',contact_backend=backend,
        contact_fixed_seed=1,calibration_decisions=128,batch=8,nodes=8,history=2,encoder_dim=4,
        hidden=[16,16],observer_updates=1,operator_updates=1,probe_count=2)
    values.update(extra)
    return make_config(case,'contact',overrides=values)


def synthetic(c,goal_episodes):
    n=observation_dim(c);replay=Replay(128,n,c['history'],include_requested_hold=True);outcomes=[]
    for episode in range(10):
        goal=episode in goal_episodes
        replay.add(dict(obs=np.zeros(n),next_obs=np.zeros(n),prev_action=np.zeros(6),action=np.zeros(6),
            prev_reward=0.,prev_h=0.,r_src=1. if goal else -1.,R=5. if goal else -2.,h=.002,
            h_requested=.01,next_h=.01,terminated=1.,truncated=0.,g=3.,time=0.,next_time=.002),episode)
        outcomes.append(dict(episode=episode,success=goal,target_speed=3.,terminal_reason='goal' if goal else 'fall'))
    return replay,{},dict(path_calibration_outcomes=outcomes,acquisition_complete=True)


class SplitTests(unittest.TestCase):
    def test_rare_goals_are_kept_in_both_episode_partitions(self):
        c=csmall();replay,_,backend=synthetic(c,{0,1})
        _,oldval=replay.episode_split()
        self.assertFalse(set(replay.episodes[oldval])&{0,1})
        strata={r['episode']:(r['target_speed'],r['terminal_reason']) for r in backend['path_calibration_outcomes']}
        train,val=replay.episode_split(strata)
        self.assertTrue(set(replay.episodes[train])&{0,1});self.assertTrue(set(replay.episodes[val])&{0,1})
        self.assertFalse(set(replay.episodes[train])&set(replay.episodes[val]))
        self.assertEqual(set(train)|set(val),set(replay.indices()))
        np.testing.assert_array_equal(train,replay.episode_split(strata)[0])

    def test_singleton_outcomes_stay_in_training_and_missing_labels_fail(self):
        c=csmall();replay,_,_=synthetic(c,{0,1})
        strata={i:('goal' if i in (0,1) else 'single' if i==9 else 'fall') for i in range(10)}
        train,val=replay.episode_split(strata)
        self.assertIn(9,set(replay.episodes[train]));self.assertNotIn(9,set(replay.episodes[val]))
        del strata[2]
        with self.assertRaises(ValueError):replay.episode_split(strata)


TORCH=bool(importlib.util.find_spec('torch'))
SIM=TORCH and bool(importlib.util.find_spec('mujoco')) and bool(importlib.util.find_spec('dm_control'))


@unittest.skipUnless(TORCH,'Torch required for preparation checks')
class AdmissionTests(unittest.TestCase):
    def test_missing_goal_coverage_still_stops_before_fitting_and_saves_data(self):
        from fjbench.calibration import prepare
        c=csmall(calibration_decisions=10);replay,counters,backend=synthetic(c,{0})
        with tempfile.TemporaryDirectory() as temp,patch('fjbench.calibration.acquire',return_value=(replay,counters,backend)),patch('fjbench.calibration.InformationMap') as fitting:
            root=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'coverage failed before fitting'):prepare(c,root,'cpu')
            self.assertTrue((root/'acquired.pt').exists());self.assertFalse((root/'prepared.pt').exists())
            self.assertFalse(json.loads((root/'calibration_acquisition.json').read_text())['admission']['admitted'])
            fitting.assert_not_called()

    def test_failed_physical_probes_are_bounded_and_do_not_create_successes(self):
        from fjbench.calibration import acquire,partition_acquisition
        c=csmall(path_calibration_goal_episodes=2,path_calibration_goal_attempts=2)
        n=observation_dim(c)
        class FailedEnvironment:
            def __init__(self,*args,**kwargs):
                self.observation=np.zeros(n);self.backend={};self.contact_map=None;self.g=3.;self.base=SimpleNamespace(cur_t=0.)
                self.path_task=SimpleNamespace(reason='fall',summary=lambda trace:dict(success=False,target_speed=3.,terminal_reason='fall'))
            def hold(self):return .01
            def step(self,action):
                self.base.cur_t+=.002
                return self.observation,dict(r_src=-1.,R=-2.,h=.002,next_h=.01,terminated=True,truncated=False,physics_steps=1)
            def close(self):pass
        with patch('fjbench.calibration.Environment',FailedEnvironment):replay,counters,backend=acquire(c)
        self.assertEqual(replay.count,2);self.assertFalse(backend['acquisition_complete'])
        _,_,admission=partition_acquisition(c,replay,backend)
        self.assertFalse(admission['admitted']);self.assertEqual(admission['goal_episodes_collected'],0)


@unittest.skipUnless(SIM,'Torch and MuJoCo required for small preparation checks')
class SimulatorTests(unittest.TestCase):
    def test_small_F_J_acquisition_has_measured_goals_without_extra_budget(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from fjbench.calibration import acquire,partition_acquisition
        for backend in ('ct_terrain','mesh'):
            for case in ('F','J'):
                c=csmall(case,backend);replay,counters,backend_info=acquire(c)
                train,val,report=partition_acquisition(c,replay,backend_info)
                self.assertTrue(report['admitted'],report)
                self.assertEqual(replay.count,128);self.assertEqual(counters['decisions'],128)
                self.assertEqual(counters['calibration_goal_probe_decisions'],4)
                self.assertEqual(report['path_goal_coverage_by_task']['3.0'],dict(train=3,validation=1))
                self.assertFalse(set(replay.episodes[train])&set(replay.episodes[val]))
                goal_ids={r['episode'] for r in backend_info['path_calibration_outcomes'] if r['success']}
                for episode in goal_ids:
                    ix=np.flatnonzero(replay.episodes[:replay.size]==episode)
                    self.assertEqual(replay.arrays['terminated'][ix[-1],0],1.)
                    self.assertGreater(replay.arrays['R'][ix[-1],0],4.9)

    def test_multiple_noninteger_target_speeds_have_separate_goal_coverage(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        from fjbench.calibration import acquire,partition_acquisition
        c=csmall('J',speed_train=[2.7,3.3],speed_test=[4.1])
        replay,counters,backend=acquire(c);_,_,report=partition_acquisition(c,replay,backend)
        self.assertTrue(report['admitted'],report)
        self.assertEqual(counters['decisions'],128)
        for speed in ('2.7','3.3'):self.assertEqual(report['path_goal_coverage_by_task'][speed],dict(train=3,validation=1))

    def test_prepare_retry_reuses_measured_data_after_a_fitting_interruption(self):
        os.environ.setdefault('MUJOCO_GL','egl')
        import torch
        from fjbench.calibration import prepare,load_prepared
        torch.set_num_threads(1)
        for case in ('F','J'):
            c=csmall(case)
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                with patch('fjbench.calibration.InformationMap',side_effect=RuntimeError('synthetic fit interruption')):
                    with self.assertRaisesRegex(RuntimeError,'synthetic fit interruption'):prepare(c,root,'cpu')
                self.assertTrue((root/'acquired.pt').exists());self.assertFalse((root/'prepared.pt').exists())
                with patch('fjbench.calibration.acquire',side_effect=AssertionError('Acquisition repeated')):
                    packet=prepare(c,root,'cpu')
                self.assertTrue(packet['acquisition_reused']);self.assertEqual(packet['counters']['decisions'],128)
                self.assertTrue(packet['acquisition_admission']['admitted'])
                self.assertTrue((root/'prepared.pt').exists())
                loaded=load_prepared(root/'prepared.pt',c,'cpu');self.assertEqual(loaded[0]['config'],c)


class InstallerTests(unittest.TestCase):
    def test_installer_keeps_a_backup_in_an_existing_v2_named_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'TPHT_CT_RL_Benchmark_v2';(root/'fjbench').mkdir(parents=True)
            (root/'fjbench/environment.py').write_text('purpose initial_speed path_task')
            (root/'fjbench/models.py').write_text('with_survival')
            old=root/'fjbench/calibration.py';old.write_text('old calibration')
            result=subprocess.run([sys.executable,str(ROOT/'apply_calibration_fix.py'),'--target',str(root)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(old.read_bytes(),(ROOT/'fjbench/calibration.py').read_bytes())
            backups=list((root/'calibration_fix_backups').rglob('calibration.py'))
            self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_text(),'old calibration')


if __name__=='__main__':unittest.main()
