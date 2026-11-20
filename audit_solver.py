"""Check all finite solvers on the same frozen replay batch. No training occurs."""
import argparse
import json
from pathlib import Path
from bench.protocol import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Matched run directory containing latest.pt")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from bench.learner import Learner, Replay
    from bench.operators import torch_collocation, phase_coefficients
    root = Path(a.run)
    state = torch.load(root / "latest.pt", map_location=a.device, weights_only=False)
    j, s = state["job"], state["learner"]
    metadata = json.loads((root / "run.json").read_text())
    model = Learner(**s["dimensions"], job=j, device=a.device, entropy_weight=s["entropy_weight"])
    model.load(s, optimizers=False)
    replay = Replay(j["settings"]["replay_capacity"], **s["dimensions"], seed=7)
    replay.load(state["replay"])
    replay.rng = np.random.default_rng(7)
    torch.manual_seed(7)
    rows = []
    with torch.no_grad():
        for batch_number in range(a.batches):
            batch = replay.sample(j["settings"]["batch_size"], a.device, metadata["native_dt"])
            x, act, xp = batch["obs"], batch["actions"], batch["next_obs"]
            ap, logp = model.actor.sample(xp)
            z = model.q(x, act, target=True).min(0).values
            next_z = model.q(xp, ap, target=True).min(0).values
            discount = torch.exp(np.log(j["settings"]["gamma_native"]) * batch["dt"] / metadata["native_dt"]) * (1 - batch["terminated"])
            args = (x, act, xp, ap, batch["reward"], discount,
                    j["settings"]["alpha_rate"] * s["entropy_weight"] * logp, z, next_z, j["settings"])
            reference, _ = torch_collocation(*args, "collocation_direct")
            for method in ("picard", "ct_picard", "tpht", "tpht_damping", "collocation_direct"):
                target, stats = torch_collocation(*args, method)
                gamma = stats["contraction_bound"]
                value_error = float((target - reference).abs().max())
                rounding = 4 * torch.finfo(torch.float32).eps * max(1., float(reference.abs().max()))
                bound = stats["operator_residual"] / (1 - gamma) + rounding
                bound_pass = value_error <= bound
                energy_pass = True
                if method.startswith("tpht"):
                    omega = 0. if method == "tpht_damping" else j["settings"]["solver_omega"]
                    c, t = phase_coefficients(j["settings"]["solver_sigma"], j["settings"]["solver_damping"], omega)
                    ratio = (c*c + t*t) ** j["settings"]["solver_steps"]
                    energy_pass = abs(stats["energy_after"] - ratio * stats["energy_before"]) <= 1e-7 * max(1., stats["energy_before"])
                rows.append(dict(batch=batch_number, method=method, value_error=value_error,
                                 finite_value_bound=bound, bound_pass=bound_pass,
                                 midpoint_energy_pass=energy_pass, **stats))
                if not bound_pass or not energy_pass:
                    raise AssertionError(rows[-1])
    write_json(a.out, dict(job=j, checkpoint_decisions=state["counters"]["decisions"],
                          scope="frozen sampled operator; no global neural or control guarantee", rows=rows))
    print(f"Audited {len(rows)} frozen solver cases: {a.out}")


if __name__ == "__main__":
    main()
