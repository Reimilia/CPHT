"""Checkpoint compatibility, state copies, and bounded acceleration fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from physical_fj.mechanics import full_mass


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Non-finite JSON value')
    return value


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False) + '\n')


def read_checkpoint(path):
    """Read the supplied legacy NumPy-containing checkpoint with restricted Torch loading."""
    import torch
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:
        from numpy.core.multiarray import _reconstruct
    safe = [_reconstruct, np.ndarray, np.dtype]
    safe += [type(np.dtype(d)) for d in ('float32', 'float64', 'int32', 'int64', 'bool')]
    with torch.serialization.safe_globals(safe):
        saved = torch.load(path, weights_only=True, map_location='cpu')
    if saved.get('schema') != 'physical_first_v1' or saved.get('method') != 'pure_cpht':
        raise ValueError('This comparison needs a physical_first_v1 pure_cpht final.pt')
    saved['_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return saved


def copied(model, data):
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    return scratch


def integration_state(model, data):
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, signature))
    mujoco.mj_getState(model, data, state, signature)
    return signature, state


def step_native(model, data, legacy=True):
    """Match dm_control's Euler split-step convention, including synced kinematics."""
    if legacy:
        mujoco.mj_step2(model, data)
        mujoco.mj_step1(model, data)
    else:
        mujoco.mj_step(model, data)


def bounded_ls(jacobian, offset, desired, ridge, low=-1., high=1.):
    weights = np.array([1., 4., 4., .15, .15, .15, .15, .15, .15])
    root = np.sqrt(weights)
    matrix = np.vstack([root[:, None]*jacobian, np.sqrt(ridge)*np.eye(6)])
    rhs = np.r_[root*(desired-offset), np.zeros(6)]
    result = lsq_linear(matrix, rhs, bounds=(np.broadcast_to(low, (6,)), np.broadcast_to(high, (6,))),
                        method='bvls', tol=1e-10, max_iter=100)
    if not result.success:
        raise RuntimeError('Bounded least squares did not converge: '+result.message)
    gradient = matrix.T@(matrix@result.x-rhs)
    projected = result.x-np.clip(result.x-gradient, low, high)
    return result.x, dict(ls_projected_gradient_inf=float(np.max(np.abs(projected))),
                          weighted_acceleration_residual=float(np.linalg.norm(root*(jacobian@result.x+offset-desired))),
                          ls_iterations=result.nit)


def acceleration_map(model, data, actuator_matrix):
    """Freeze the measured current constraint port; do not freeze the mass matrix in time."""
    mass = full_mass(model, data)
    jacobian = np.linalg.solve(mass, actuator_matrix)
    # qacc already includes passive, bias, external, and current constraint forces.
    offset = data.qacc-jacobian@data.ctrl
    return jacobian, offset
