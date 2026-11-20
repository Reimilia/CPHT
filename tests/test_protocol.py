import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bench.protocol import TASKS, source_root
from bench.env import entropy_weight
from bench.operators import finite_step, phase_step


class ProtocolTests(unittest.TestCase):




    def test_entropy_uses_native_time_units(self):
        gamma, native, dt = .98, .01, TASKS["cheetah-run"]["dt"]
        beta = -np.log(gamma)
        expected = -np.expm1(-beta * dt / native) / beta
        self.assertAlmostEqual(entropy_weight("cheetah-run", native, gamma, "regular"), expected)
        # Unit conversion leaves the dimensionless regular weight unchanged.
        self.assertGreater(entropy_weight("cheetah-run", native, gamma, "irregular"), 0)

    def test_source_is_pinned(self):
        self.assertTrue((source_root() / "algorithms/ct_sac.py").is_file())


class SolverTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.p = rng.uniform(size=(19, 19))
        self.p /= self.p.sum(1, keepdims=True)
        self.gamma = rng.uniform(.8, .99, 19)
        self.gamma[0] = 0  # true terminal, row stays a valid probability vector
        self.b, self.z = rng.normal(size=(2, 19))
        self.a = np.eye(19) - self.gamma[:, None] * self.p

    def test_positive_operator_is_sup_norm_contraction(self):
        z2 = self.z + np.arange(19)
        delta = self.gamma * (self.p @ (self.z - z2))
        self.assertLessEqual(np.max(abs(delta)), self.gamma.max() * np.max(abs(self.z - z2)) + 1e-12)

    def test_midpoint_energy_identity(self):
        e = self.a @ self.z - self.b
        rho = np.linspace(-1., 1., 19)
        ep, rp = phase_step(e, rho, .5, 1., 3.)
        before, after = .5 * (e@e + rho@rho), .5 * (ep@ep + rp@rp)
        midpoint = ((e + ep) / 2, (rho + rp) / 2)
        self.assertAlmostEqual(after - before, -.5 * sum(v@v for v in midpoint), places=11)
        self.assertLess(after, before)

    def test_inverse_recovers_prescribed_residual(self):
        target, info = finite_step(self.z, self.b, self.gamma, self.p, "tpht")
        expected_error, _ = phase_step(self.a @ self.z - self.b, np.zeros(19), .5, 1., 3.)
        np.testing.assert_allclose(self.a @ target - self.b, expected_error, atol=1e-12)
        np.testing.assert_allclose(info["conjugate_momentum"], self.a.T @ info["rho"])

    def test_every_method_obeys_finite_residual_value_bound(self):
        fixed = np.linalg.solve(self.a, self.b)
        for method in ("picard", "ct_picard", "tpht", "tpht_damping", "collocation_direct"):
            z, _ = finite_step(self.z, self.b, self.gamma, self.p, method)
            bound = np.max(abs(self.a @ z - self.b)) / (1 - self.gamma.max())
            self.assertLessEqual(np.max(abs(z - fixed)), bound + 1e-11)

    def test_ct_style_identity(self):
        ordinary, _ = finite_step(self.z, self.b, self.gamma, self.p, "picard")
        ct, _ = finite_step(self.z, self.b, self.gamma, self.p, "ct_picard", eta=.5)
        np.testing.assert_allclose(ct, .5 * (ordinary + self.z))

    def test_invalid_probability_rows_fail(self):
        with self.assertRaises(ValueError):
            finite_step(self.z, self.b, self.gamma, self.p * 2, "tpht")


if __name__ == "__main__":
    unittest.main()
