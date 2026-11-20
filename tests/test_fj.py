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
from fjbench.data import Replay,initial_history,next_history
from fjbench.environment import ou_step
from fjbench.finite import backup,recover,solve,residual_vjp,entropy


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

    def test_entropy_units_and_actor_identity(self):
        c=make_config();h=.024;reference=c["reference_time"]
        self.assertAlmostEqual(entropy(h,c),c["alpha_reference"]*h/reference)
        c2=make_config(overrides={"entropy_convention":"discounted"})
        self.assertLess(entropy(h,c2),entropy(h,c))
        value,rate,lp=1.3,-.7,-2.4
        q=value+h*rate;combined=value+reference*rate
        left=entropy(h,c)*lp-q
        right=h/reference*(c["alpha_reference"]*lp-combined)-(1-h/reference)*value
        self.assertAlmostEqual(left,right)

    def test_ou_noise_is_in_the_environment_with_correct_increment(self):
        a,b=np.random.default_rng(31),np.random.default_rng(31)
        expected=np.exp(-.2)*.7+.4*np.sqrt(-np.expm1(-.4)/2)*b.normal()
        self.assertAlmostEqual(ou_step(.7,1.,.4,.2,a),expected)
        self.assertAlmostEqual(ou_step(.7,1.,0.,.2,a),np.exp(-.2)*.7)

    def test_commands_are_dry_without_optional_dependencies(self):
        commands=[("prepare_fj.py","F","position"),("train_fj.py","J","wind"),("eval_fj.py","J","static")]
        for script,case,experiment in commands:
            result=subprocess.run([sys.executable,"-S",str(ROOT/script),"--case",case,"--experiment",experiment,"--dry-run"],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)["expensive_work_launched"])


class ReplayTests(unittest.TestCase):
    def test_current_history_excludes_the_successor(self):
        replay=Replay(20,2,3)
        for i in range(4):replay.add(record(i),episode=0)
        b=replay.windows([2])
        np.testing.assert_array_equal(b["window"][0,-1,:2],[2,3])
        np.testing.assert_array_equal(b["next_window"][0,-1,:2],[3,4])
        self.assertNotEqual(b["window"][0,-1,0],b["next_obs"][0,0])

    def test_episode_boundary_and_ring_overwrite_do_not_leak_history(self):
        replay=Replay(7,2,3)
        for i in range(20):replay.add(record(i),i//4)
        b=replay.windows([16%7])
        np.testing.assert_array_equal(b["window"][0,:-1],0.)
        np.testing.assert_array_equal(b["window"][0,-1,:2],[16,17])

    def test_hidden_parameters_are_rejected_as_public_fields(self):
        replay=Replay(7,2)
        r=record(1);r["hidden_mass"]=1.
        with self.assertRaises(ValueError):replay.add(r,0)

    def test_validation_exclusion_survives_recovery_and_wrap(self):
        replay=Replay(20,2,3,11)
        for i in range(20):replay.add(record(i,g=6. if i%2 else 10.),i//4)
        replay.exclude_serials([1,2,3,4])
        self.assertFalse(set(replay.sample(500)["obs"][:,0])&{1,2,3,4})
        recovered=Replay.restore(replay.state())
        np.testing.assert_array_equal(replay.sample(32)["obs"],recovered.sample(32)["obs"])
        for i in range(20,30):recovered.add(record(i),100)
        self.assertTrue(np.all(recovered.sample(200)["obs"][:,0]>=10))


class TwinOperatorTests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(73);n,k=7,3
        self.p=r.uniform(size=(n,k,n));self.p/=self.p.sum(-1,keepdims=True)
        self.gamma=r.uniform(.9,.997,(n,k));self.gamma[0]=0
        self.reward=r.normal(size=(n,k));self.ent=r.normal(size=(n,k))*.1
        self.q=r.normal(size=(2,n,k));self.c=make_config()

    def test_recovery_changes_both_free_heads(self):
        e=np.random.default_rng(2).normal(size=self.q.shape)
        q=recover(e,self.reward,self.gamma,self.p,self.ent)
        np.testing.assert_allclose(q-backup(q,self.reward,self.gamma,self.p,self.ent),e,atol=2e-13)
        self.assertGreater(np.max(abs(q[0]-q[1])),.1)

    def test_twin_min_operator_is_a_sup_norm_contraction(self):
        other=self.q+.4*np.random.default_rng(6).normal(size=self.q.shape)
        error=np.max(abs(backup(self.q,self.reward,self.gamma,self.p,self.ent)-backup(other,self.reward,self.gamma,self.p,self.ent)))
        self.assertLessEqual(error,self.gamma.max()*np.max(abs(self.q-other))+1e-12)

    def test_full_terminal_boundary_removes_future_entropy(self):
        out=backup(self.q,self.reward,self.gamma,self.p,self.ent)
        np.testing.assert_allclose(out[:,0],np.stack([self.reward[0],self.reward[0]]))

    def test_phase_storage_and_value_bound(self):
        fixed=recover(np.zeros_like(self.q),self.reward,self.gamma,self.p,self.ent)
        for method in ("interval_euler","ph_sac","ph_direct"):
            q,rho,stats=solve(self.q,self.reward,self.gamma,self.p,self.ent,self.c,method)
            self.assertLessEqual(np.max(abs(q-fixed)),stats["finite_value_bound"]+1e-11)
            if method=="ph_sac":
                self.assertLess(stats["energy_after"],stats["energy_before"])
                self.assertLess(stats["inverse_residual"],1e-10)
                self.assertLess(stats["midpoint_balance_defect"],1e-11)

    def test_cotangent_vjp_matches_a_directional_derivative(self):
        rng=np.random.default_rng(84)
        q=self.q.copy();q[1]=q[0]+2.
        rho=rng.normal(size=q.shape);direction=rng.normal(size=q.shape)
        eps=1e-6
        def residual(x):return x-backup(x,self.reward,self.gamma,self.p,self.ent)
        derivative=(residual(q+eps*direction)-residual(q-eps*direction))/(2*eps)
        self.assertAlmostEqual(np.sum(rho*derivative),np.sum(residual_vjp(q,rho,self.gamma,self.p)*direction),places=7)

    def test_operator_change_residual_identity(self):
        changed=self.reward+.2
        old=self.q-backup(self.q,self.reward,self.gamma,self.p,self.ent)
        new=self.q-backup(self.q,changed,self.gamma,self.p,self.ent)
        np.testing.assert_allclose(new,old-(backup(self.q,changed,self.gamma,self.p,self.ent)-backup(self.q,self.reward,self.gamma,self.p,self.ent)))


@unittest.skipUnless(importlib.util.find_spec("torch"),"Torch runtime checks remain optional")
class NeuralCaseTests(unittest.TestCase):
    def test_positive_conditioned_rows_and_all_adapters(self):
        import torch
        from fjbench.models import InformationMap,OperatorModel,tensors
        from fjbench.agent import Agent
        from fjbench.config import METHODS
        c=make_config(overrides={"batch":8,"history":3,"nodes":6,"action_samples":2,
                                "hidden":[16,16],"fit_steps":1})
        replay=Replay(20,17,3)
        for i in range(20):
            r=record(i);r["obs"]=np.full(17,i/100);r["next_obs"]=np.full(17,(i+1)/100)
            replay.add(r,0)
        nodes=tensors(replay.windows(range(6)),"cpu")
        info=InformationMap(c,np.zeros(25),np.ones(25)).requires_grad_(False)
        ni=info.encode_batch(nodes)
        model=OperatorModel(info.feature_dim,{"g":nodes["g"].numpy()},c,[1.,1.])
        p,r=model.predict(ni,torch.zeros(6,6),nodes["g"])
        np.testing.assert_allclose(p.detach().sum(1).numpy(),1,atol=1e-6)
        batch=tensors(replay.sample(8),"cpu")
        for method in METHODS:
            agent=Agent(c,method,info,model,nodes,ni,"cpu")
            result=agent.update(batch)
            self.assertTrue(np.isfinite(result["critic_loss"]))
            self.assertGreaterEqual(result["critic_steps"],1)


if __name__=="__main__":unittest.main()
