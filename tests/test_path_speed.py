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







SIM=importlib.util.find_spec('mujoco') and importlib.util.find_spec('dm_control')


@unittest.skipUnless(SIM,'MuJoCo task checks')
class EnvironmentTests(unittest.TestCase):

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
