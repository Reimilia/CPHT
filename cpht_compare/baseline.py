"""Model-free CT-SAC baseline under the same calibration, sensors and budget.

The baseline reuses the shared calibration artifact of the pH methods: its
replay starts from the same calibration transitions, it acts on the same
PhysicalEnvironment measurements (state, state_belief or sensed_belief) and
it spends the same total decision budget. Its critic uses the source CT-SAC
rate target (paper eq. 2.26). It learns no physical model.
"""
from pathlib import Path
import time

import numpy as np
import torch

from bench.protocol import digest, write_json
from bench.training import atomic_torch_save

METHOD = "ct_sac"


def train(profile, job, s, c, artifact, run, device="cpu", resume=False):
    """Train CT-SAC with physical_fj.train and store a comparison checkpoint."""
    from physical_fj.train import train as train_sac
    from physical_fj.prepare import load as load_calibration
    from pure_cpht.learning import prepare_artifact

    run = Path(run)
    final = run / "final.pt"
    if final.exists() and not resume:
        return dict(status="existing_complete_checkpoint", checkpoint=str(final))
    learner_dir = run / "ct_sac"
    counters = train_sac(s, c, METHOD, artifact, learner_dir, device, resume=resume)
    if counters["decisions"] < s["total_decisions"]:
        return dict(complete=False, counters=counters, checkpoint=None)
    learned = torch.load(learner_dir / "final.pt", map_location="cpu", weights_only=False)
    calibration, _, norm = load_calibration(artifact)
    shared_model = prepare_artifact(artifact, s)["model"]
    from environment_belief.config import contract
    packet = dict(
        schema=profile["schema"], profile=profile, job=dict(job), settings=s, control=c,
        calibration=calibration, normalization=norm,
        # Identity of the shared calibration artifact. CT-SAC does not use the model.
        model_sha256=digest(shared_model), information_contract=contract(s),
        sac=dict(actor=learned["agent"]["actor"], dimension=learned["dimension"],
                 feature_kind=learned["feature_kind"], hidden=s["hidden"]),
        counters=dict(learned["counters"],
                      physics_steps=learned["counters"]["policy_physics_steps"],
                      physical_seconds=learned["counters"]["policy_physics_steps"] * c["physics_dt"]),
        optimization_scope="model-free CT-SAC actor and twin critics")
    atomic_torch_save(packet, final)
    result = dict(complete=True, counters=packet["counters"], checkpoint=str(final))
    write_json(run / "status.json", result)
    return result


class Policy:
    """Deterministic CT-SAC actor with the evaluator's controller interface."""

    def __init__(self, saved, device="cpu"):
        from bench.learner import Actor
        sac = saved["sac"]
        if sac["feature_kind"] != "raw":
            raise ValueError("The comparison baseline acts on the raw measurement vector")
        self.actor = Actor(sac["dimension"], 6, sac["hidden"]).to(device)
        self.actor.load_state_dict(sac["actor"])
        self.actor.eval().requires_grad_(False)
        self.device = device
        self.norm = saved["normalization"]
        self.coefficients = saved["calibration"]["coefficients"]
        self.diagnostics, self.failure_counts = {}, {}
        self.plans = self.predicted_steps = 0

    def act(self, packet, target, hold, time_now):
        from physical_fj.train import normalized
        started = time.perf_counter()
        observation = normalized(packet, "raw", self.coefficients, self.norm)
        with torch.no_grad():
            tensor = torch.as_tensor(observation, device=self.device).reshape(1, -1)
            action = self.actor.sample(tensor, True)[0][0].cpu().numpy()
        self.diagnostics = dict(action_saturation=float(np.mean(np.abs(action) > .99)),
                                decision_seconds=time.perf_counter() - started,
                                actor_updates=0, critic_updates=0)
        return action

    def prediction(self, *_):
        return None
