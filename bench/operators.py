"""Exact finite-operator arithmetic. Tests require only NumPy."""
from __future__ import annotations
import numpy as np


def phase_coefficients(sigma, damping, omega):
    if sigma <= 0 or damping <= 0:
        raise ValueError("Positive solver time and damping are required")
    a, b = sigma * damping / 2, sigma * omega / 2
    denominator = (1 + a) ** 2 + b ** 2
    c, s = (1 - a*a - b*b) / denominator, 2*b / denominator
    return c, s


def phase_step(error, rho, sigma, damping, omega):
    c, s = phase_coefficients(sigma, damping, omega)
    return c * error + s * rho, -s * error + c * rho


def finite_step(z, b, discount, transition, method, eta=.5, sigma=.5, damping=1., omega=3.):
    """Reference implementation for the same arithmetic used by the Torch trainer."""
    z, b, discount, transition = map(np.asarray, (z, b, discount, transition))
    if np.any(discount < 0) or np.any(discount >= 1):
        raise ValueError("Discounts must lie in [0,1)")
    if np.any(transition < 0) or not np.allclose(transition.sum(1), 1):
        raise ValueError("Transition rows must be probability vectors")
    a = np.eye(len(z)) - discount[:, None] * transition
    error = a @ z - b
    rho = np.zeros_like(z)
    if method in ("picard", "ct_picard"):
        out = z - (1. if method == "picard" else eta) * error
    elif method == "collocation_direct":
        out = np.linalg.solve(a, b)
    elif method in ("tpht", "tpht_damping"):
        enew, rho = phase_step(error, rho, sigma, damping, 0. if method == "tpht_damping" else omega)
        out = np.linalg.solve(a, b + enew)
    else:
        raise ValueError(method)
    return out, dict(energy_before=float(error @ error / 2),
                     energy_after=float((np.linalg.norm(a @ out - b)**2 + rho @ rho) / 2),
                     rho=rho, conjugate_momentum=a.T @ rho)


def torch_collocation(obs, actions, next_obs, next_actions, reward, discount,
                      entropy, target_z, network_next_z, cfg, method, collect_stats=True):
    """Freeze one sampled soft policy operator and transport its critic values.

    z_i is a soft Q value at the replay anchor (s_i,a_i). Positive kernel rows
    interpolate Q at (s'_i,pi(s'_i)). The operator is affine for this update.
    """
    import torch
    anchors = torch.cat((obs, actions), 1).double()
    queries = torch.cat((next_obs, next_actions), 1).double()
    centre = anchors.mean(0)
    scale = anchors.std(0, unbiased=False).clamp_min(.1)
    # Separate state and action blocks, so a large observation dimension does
    # not erase the action distance. The metric is shared by every solver.
    scale[:obs.shape[1]] *= obs.shape[1] ** .5
    scale[obs.shape[1]:] *= actions.shape[1] ** .5
    distance = torch.cdist((queries - centre) / scale, (anchors - centre) / scale) ** 2
    count = min(int(cfg["knn"]), len(anchors))
    values, indices = distance.topk(count, largest=False, dim=1)
    weights = torch.softmax(-values / cfg["kernel_temperature"], dim=1)
    transition = torch.zeros_like(distance).scatter(1, indices, weights)
    gamma = discount.flatten().double()
    b = reward.flatten().double() - gamma * entropy.flatten().double()
    z = target_z.flatten().double()
    a = torch.eye(len(z), device=z.device, dtype=z.dtype) - gamma[:, None] * transition
    error = a @ z - b
    rho = torch.zeros_like(z)
    solves, matvecs = 0, 1
    ephase = error
    if method in ("picard", "ct_picard"):
        result = z - (1. if method == "picard" else cfg["picard_eta"]) * error
    elif method == "collocation_direct":
        result = torch.linalg.solve(a, b)
        solves = 1
    else:
        omega = 0. if method == "tpht_damping" else cfg["solver_omega"]
        c, s = phase_coefficients(cfg["solver_sigma"], cfg["solver_damping"], omega)
        for _ in range(cfg["solver_steps"]):
            ephase, rho = c * ephase + s * rho, -s * ephase + c * rho
        # The frozen policy operator is affine. Solve e(z)=ephase directly.
        # This cost is recorded and compared with the direct-solve baseline.
        result = torch.linalg.solve(a, b + ephase)
        solves = 1
    if not torch.isfinite(result).all():
        raise FloatingPointError("The finite solver produced non-finite targets")
    stats = dict(operator_matvecs=matvecs, linear_solves=solves, knn_queries=len(z),
                 distance_pairs=len(z) ** 2)
    if not collect_stats:
        return result.float().reshape(-1, 1), stats
    actual_error = a @ result - b
    inverse_defect = (actual_error - ephase).abs().max() if method.startswith("tpht") else z.new_tensor(0.)
    # NN successor predictions give an independent projection-defect diagnostic.
    projection = gamma * (transition @ z - network_next_z.flatten().double())
    stats["operator_matvecs"] += 3
    stats.update(energy_before=float(.5 * error.square().sum()),
                 energy_after=float(.5 * (actual_error.square().sum() + rho.square().sum())),
                 inverse_defect=float(inverse_defect), operator_residual=float(actual_error.abs().max()),
                 projection_defect=float(projection.abs().max()), nearest_distance=float(values[:, 0].mean()),
                 conjugate_momentum_norm=float(torch.linalg.vector_norm(a.T @ rho)),
                 contraction_bound=float(gamma.max()))
    return result.float().reshape(-1, 1), stats
