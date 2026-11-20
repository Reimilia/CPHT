import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from bench.protocol import ROOT
from fjbench.config import make_config,calibration_key
from fjbench.environment import ou_step


def record(i,g=10.):
    return dict(obs=np.array([i,i+1]),next_obs=np.array([i+1,i+2]),prev_action=np.zeros(6),action=np.ones(6),
        prev_reward=0.,prev_h=.01,r_src=float(i),R=.1,h=.01,next_h=.01,terminated=0.,truncated=0.,
        g=g,time=i*.01,next_time=(i+1)*.01)


class CaseTests(unittest.TestCase):
    def test_four_experiments_have_the_declared_sensors_and_regimes(self):
        self.assertEqual(make_config("F","native")["sensor"],"native")
        self.assertEqual(make_config("F","position")["sensor"],"position")
        self.assertEqual(make_config("J","static")["wind_sigma"],0)
        self.assertGreater(make_config("J","wind")["wind_sigma"],0)
        with self.assertRaises(ValueError):make_config("F",overrides={"load_train":[1.]})

    def test_task_and_regime_holdouts_are_separate(self):
        c=make_config("J","wind")
        for name in ("speed","friction","load"):
            self.assertFalse(set(c[name+"_train"])&set(c[name+"_test"]))

    def test_solver_settings_share_calibration_but_sensor_changes_do_not(self):
        c=make_config()
        other=make_config(overrides={"omega":4.,"phase_steps":7})
        self.assertEqual(calibration_key(c),calibration_key(other))
        self.assertNotEqual(calibration_key(c),calibration_key(make_config("F","position")))


    def test_ou_noise_is_in_the_environment_with_correct_increment(self):
        a,b=np.random.default_rng(31),np.random.default_rng(31)
        expected=np.exp(-.2)*.7+.4*np.sqrt(-np.expm1(-.4)/2)*b.normal()
        self.assertAlmostEqual(ou_step(.7,1.,.4,.2,a),expected)
        self.assertAlmostEqual(ou_step(.7,1.,0.,.2,a),np.exp(-.2)*.7)









if __name__=="__main__":unittest.main()
