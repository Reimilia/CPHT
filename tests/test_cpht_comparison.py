"""Numerical and causal checks for the new comparison, with bounded workloads."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

RUNTIME = all(importlib.util.find_spec(n) for n in ("torch", "mujoco", "dm_control", "gymnasium"))

class ConfigurationTests(unittest.TestCase):
    def test_plan_is_dry_and_has_five_methods_and_shared_calibrations(self):
        from cpht_compare.config import load, plan
        p = load("experiments/smoke.json")
        jobs = plan(p, "runs/test", "artifacts/test")
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({x["artifact"] for x in jobs}), 2)
        for case in ("F", "J"):
            selected = [x for x in jobs if x["case"] == case]
            self.assertEqual({x["method"] for x in selected},
                             {"cem", "shooting", "selector", "selector_value", "ct_sac"})
            self.assertEqual(len({x["calibration_decisions"] for x in selected}), 1)
        result = subprocess.run(
            [sys.executable, "-m", "cpht_compare", "all", "--dry-run",
             "--profile", "experiments/smoke.json"],
            capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["expensive_training_executed"])

    def test_baseline_is_trained_once_per_mode_case_seed(self):
        from cpht_compare.config import load, jobs, paths
        p = load("experiments/full.json")
        baseline = [j for j in jobs(p) if j["method"] == "ct_sac"]
        self.assertEqual(len(baseline), len(p["information_modes"]) * len(p["cases"]) * len(p["seeds"]))
        self.assertTrue(all("model_free" in str(paths(p, j)[3]) for j in baseline))
        self.assertFalse(p["learning"]["hold_disclosure"])

    def test_report_censors_incomplete_course_and_rejects_unpaired_truth(self):
        from cpht_compare.report import seed_summary, validate_pairing
        from copy import deepcopy
        row = dict(budget_censored=True, error=None, physical_seconds=.08,
                   predictions=[], controller_trace=[], failure_flags=["evaluation_budget_censored"],
                   episode_seed=1, target_speed=3, audit_truth={"contact_map_sha256": "same"},
                   predictive_substeps=8, discounted_task_return=-1,
                   path_metrics=dict(success=False, speed_rmse=2.7, completed_fraction=.001))
        a = dict(job=dict(mode="state_belief", case="F", horizon=.02, seed=0, method="cem"),
                 test_case="F", clock="fixed", condition="nominal", rows=[row],
                 calibration_id="same", model_sha256="same", evaluation_sensing={},
                 task={}, information_contract={}, train_counters=dict(physical_seconds=.04, physics_steps=20))
        self.assertIsNone(seed_summary(a)["success_rate"])
        b = deepcopy(a)
        b["job"]["method"] = "shooting"
        self.assertEqual(validate_pairing([a, b]), 1)
        b["rows"][0]["audit_truth"]["contact_map_sha256"] = "other"
        with self.assertRaises(ValueError):
            validate_pairing([a, b])

@unittest.skipUnless(RUNTIME, "Torch and MuJoCo runtime required")
class DifferentiableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from physical_fj.config import settings, control
        from physical_fj.mechanics import NominalRobot
        from pure_cpht.mechanics import public_geometry, Robot
        from pure_cpht.config import options
        from cpht_compare.config import DEFAULTS
        torch.set_num_threads(1)
        s = settings("F", pure={}, sensing=dict(mode="state_belief"))
        cls.task = control(s)
        nominal = NominalRobot(s)
        geometry = public_geometry(nominal)
        robot = Robot(geometry)
        cls.model = dict(geometry=geometry, parameters=robot.parameters.tolist(),
                         contact=[2500., 100., .08, .6])
        cls.options = options(dict(horizon_steps=3, substeps=1, candidates=4,
                                   elites=2, iterations=1, contact_iterations=4))
        cls.optimization = dict(DEFAULTS["optimization"], selector_hidden=[8],
                                batch_size=2, critic_epochs=1, shooting_steps=2)

    def controller(self, method="shooting", double=False):
        import torch
        from cpht_compare.controller import ComparisonController
        ctl = ComparisonController(self.model, self.options, [0.] * 6, self.task,
                                   self.optimization, .03, method,
                                   dict(raw_mean=np.zeros(60), raw_scale=np.ones(60)))
        if double:
            from pure_cpht.mechanics import Robot
            from pure_cpht.contact import Contact
            from cpht_compare.rollout import Imagination
            ctl.robot = Robot(self.model["geometry"], self.model["parameters"], dtype=torch.float64)
            ctl.contact = Contact(ctl.robot, self.model["contact"], self.options)
            ctl.low, ctl.high, ctl.mean = ctl.low.double(), ctl.high.double(), ctl.mean.double()
            ctl.engine = Imagination(ctl, self.task, self.optimization, .03)
        return ctl

    def packet(self):
        # Airborne state avoids contact-mode ambiguity in the smooth gradient audit.
        q = np.array([0., .3, .05, .1, -.1, .02, -.2, .1, -.05])
        raw = np.r_[q, np.zeros(9), np.zeros(9), [1., 1., .3, 1.], np.zeros(29)].astype(np.float32)
        raw[46] = .6
        x = np.array([-1., -.5, 0., .5, 1., 2., 3.])
        return dict(raw=raw, scan=np.c_[x, np.zeros(7), np.zeros(7)],
                    environment=dict(friction=.6))

    def context(self, ctl):
        from cpht_compare.rollout import snapshot, stack_snapshots
        row = snapshot(self.packet(), .2, 3., .01, points=len(ctl.robot.point_body))
        return stack_snapshots([row], ctl.robot)

    def test_rollout_directional_gradient_and_model_frozen(self):
        import torch
        ctl = self.controller(double=True)
        context = self.context(ctl)
        theta = ctl.mean[None].clone().requires_grad_()
        before = ctl.robot.parameters.clone()
        result = ctl.engine.rollout(context, parameters=theta)
        derivative = torch.autograd.grad(result.objective.mean(), theta)[0]
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        self.assertGreater(float(derivative.norm()), 1e-8)
        direction = torch.linspace(-1, 1, 19, dtype=torch.float64)[None]
        direction /= direction.norm()
        analytic = float((derivative * direction).sum())
        epsilon = 1e-5
        with torch.no_grad():
            plus = ctl.engine.rollout(context, parameters=theta + epsilon * direction).objective.mean()
            minus = ctl.engine.rollout(context, parameters=theta - epsilon * direction).objective.mean()
        self.assertAlmostEqual(analytic, float((plus - minus) / (2 * epsilon)), delta=2e-5)
        self.assertTrue(torch.equal(before, ctl.robot.parameters))
        self.assertEqual(tuple(result.raw.shape), (4, 1, 60))
        self.assertAlmostEqual(float(result.physical_durations.sum()), .03)

    def test_variable_hold_and_actual_six_input_bounds(self):
        import torch
        ctl = self.controller()
        context = self.context(ctl)
        context["first_hold"][:] = .03
        result = ctl.engine.rollout(context, parameters=ctl.mean[None])
        self.assertAlmostEqual(float(result.physical_durations.sum()), .05, places=6)
        self.assertLessEqual(float(result.actions.abs().max()), 1.)
        self.assertEqual(tuple(result.actions.shape), (3, 1, 6))

    def test_selector_and_value_updates_preserve_model_and_use_no_simulator(self):
        import torch
        from unittest.mock import patch
        from cpht_compare.learning import Replay, Trainer
        ctl = self.controller("selector_value")
        trainer = Trainer(ctl, self.optimization)
        bank = Replay(8, 0)
        context = self.context(ctl)
        bank.add({k: v[0].numpy() for k, v in context.items()})
        before_model = ctl.robot.parameters.clone()
        before_selector = [x.clone() for x in ctl.selector.parameters()]
        with patch("mujoco.mj_forward", side_effect=AssertionError("future simulator read")):
            metrics = trainer.update(ctl, bank)
        self.assertFalse(metrics["skipped"])
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(trainer.value_updates, 1)
        self.assertTrue(torch.equal(before_model, ctl.robot.parameters))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_selector, ctl.selector.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in ctl.value.parameters()))

    def test_snapshot_ignores_audit_truth_and_scalar_mu_stays_causal(self):
        from cpht_compare.rollout import snapshot
        ctl = self.controller()
        p = self.packet()
        first = snapshot(p, 0, 3, .01, points=len(ctl.robot.point_body))
        p["audit_truth"] = dict(friction=1000, future_sensor=[1e9], map_id="private")
        second = snapshot(p, 0, 3, .01, points=len(ctl.robot.point_body))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertAlmostEqual(float(first["mu"]), .6)

    def test_td_lambda_uses_real_interval_discounts_and_stops_at_terminal(self):
        import torch
        from cpht_compare.networks import lambda_targets
        rewards = torch.tensor([[1.], [2.], [3.]])
        discounts = torch.tensor([[.9], [.8], [.7]])
        continuation = torch.tensor([[.5], [0.], [1.]])
        values = torch.tensor([[10.], [20.], [30.], [40.]])
        np.testing.assert_allclose(lambda_targets(rewards, discounts, continuation, values, 0).numpy().ravel(),
                                   [10., 2., 31.])
        np.testing.assert_allclose(lambda_targets(rewards, discounts, continuation, values, 1).numpy().ravel(),
                                   [1.9, 2., 31.], rtol=1e-6)


@unittest.skipUnless(RUNTIME, "Torch and MuJoCo runtime required")
class UnifiedPipelineTests(unittest.TestCase):
    """CT-SAC and a pH method share calibration, sensors and paired evaluation."""

    def test_hidden_hold_and_paired_baseline_end_to_end(self):
        import tempfile
        import numpy as np
        from cpht_compare import config, pipeline
        from cpht_compare.report import collect
        from physical_fj.mechanics import PhysicalEnvironment
        profile = config.load("experiments/smoke.json")
        profile.update(cases=["F"], methods=["cem", "ct_sac"], information_modes=["sensed_belief"])
        profile["evaluation"].update(clocks=["irregular"], conditions=["nominal"])
        with tempfile.TemporaryDirectory() as root:
            runs, artifacts = f"{root}/runs", f"{root}/artifacts"
            job_list = list(config.jobs(profile))
            pipeline.preparation(profile, job_list[0], runs, artifacts)
            s, c, _, _ = config.paths(profile, job_list[0], runs, artifacts)
            env = PhysicalEnvironment(c, s, 3, clock="irregular", purpose="evaluation")
            try:
                seen, requested = set(), set()
                for _ in range(30):
                    seen.add(round(float(env.current["raw"][30]) * c["reference_time"], 6))
                    requested.add(round(env.hold(), 6))
                    env.step(np.zeros(6, np.float32))
            finally:
                env.close()
            self.assertEqual(seen, {c["fixed_hold"]})
            self.assertGreater(len(requested), 1)
            for job in job_list:
                result = pipeline.training(profile, job, runs, artifacts)
                self.assertTrue(result["complete"])
                run = config.paths(profile, job, runs, artifacts)[3]
                report = pipeline.evaluate(run / "final.pt", "F", "irregular", "nominal",
                                           evaluation=profile["evaluation"])
                self.assertFalse(report["hold_disclosure"])
                self.assertTrue(all(row["error"] is None for row in report["rows"]))
            summary = collect(profile, runs, artifacts, f"{root}/report.json")
            self.assertEqual((summary["report_count"], summary["missing_count"], summary["paired_groups"]), (2, 0, 1))

if __name__ == "__main__":
    unittest.main()
