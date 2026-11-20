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


