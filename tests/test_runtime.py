"""Optional, small runtime checks. These tests do not collect training rollouts."""
import importlib.util
import unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Torch is optional for the dependency-light protocol checks")
class TorchTests(unittest.TestCase):
    def test_torch_solver_matches_reference_for_identity_kernel(self):
        import torch
        from bench.protocol import make_job
        from bench.operators import finite_step, torch_collocation
        c = make_job("matched", "cheetah-run", "tpht", 0)["settings"]
        c["knn"] = 1
        x = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        act = torch.linspace(-.8, .8, 7).reshape(-1, 1)
        z = torch.linspace(-2., 3., 7).reshape(-1, 1)
        reward = torch.arange(7, dtype=torch.float32).reshape(-1, 1) / 10
        gamma = torch.full((7, 1), .9)
        entropy = torch.full((7, 1), -.2)
        for method in ("picard", "ct_picard", "tpht", "tpht_damping", "collocation_direct"):
            out, stats = torch_collocation(x, act, x, act, reward, gamma, entropy, z, z, c, method)
            b = (reward - gamma * entropy).numpy().flatten()
            reference, _ = finite_step(z.numpy().flatten(), b, gamma.numpy().flatten(), np.eye(7), method)
            np.testing.assert_allclose(out.numpy().flatten(), reference, rtol=2e-6, atol=2e-6)
            self.assertTrue(np.isfinite(stats["projection_defect"]))

    def test_one_update_of_each_neural_adapter(self):
        import torch
        from bench.protocol import make_job, MATCHED
        from bench.learner import Learner
        torch.manual_seed(9)
        for method in MATCHED:
            job = make_job("matched", "cheetah-run", method, 0,
                           settings=dict(batch_size=8, warmup=8, hidden=[16, 16], knn=3))
            learner = Learner(5, 2, job, "cpu", .5)
            batch = dict(obs=torch.randn(8, 5), next_obs=torch.randn(8, 5),
                         actions=torch.rand(8, 2) * 2 - 1, reward=torch.rand(8, 1),
                         endpoint_rate=torch.rand(8, 1), dt=torch.full((8, 1), .01),
                         native_dt=.01, terminated=torch.zeros(8, 1))
            before = [p.detach().clone() for p in learner.critics.parameters()]
            stats = learner.update(batch, diagnostics=True)
            self.assertTrue(np.isfinite(stats["critic_loss"]))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.critics.parameters())))
            self.assertEqual(learner.updates, 1)

    def test_replay_state_round_trip(self):
        from bench.learner import Replay
        replay = Replay(11, 3, 2, 8)
        for i in range(17):
            replay.add(obs=np.full(3, i), next_obs=np.full(3, i+1), actions=np.zeros(2),
                       reward=float(i), endpoint_rate=.5, dt=.01, terminated=0.)
        restored = Replay(11, 3, 2, 91)
        restored.load(replay.state())
        first = replay.sample(8, "cpu", .01)
        second = restored.sample(8, "cpu", .01)
        np.testing.assert_array_equal(first["obs"].numpy(), second["obs"].numpy())


if __name__ == "__main__":
    unittest.main()
