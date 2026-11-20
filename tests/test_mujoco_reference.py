"""Underactuation, bounded fitting, and real-transition replay regressions."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from mujoco_reference.common import bounded_ls


def test_box_fitting_is_not_unconstrained_inverse_then_clipping():
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(9, 6))
    matrix[:, 1] += .9*matrix[:, 0]
    target = 20*rng.normal(size=9)
    action, diagnostics = bounded_ls(matrix, np.zeros(9), target, .01)
    weights = np.array([1., 4., 4., .15, .15, .15, .15, .15, .15])
    unconstrained = np.linalg.solve(matrix.T@(weights[:, None]*matrix)+.01*np.eye(6), matrix.T@(weights*target))
    clipped = unconstrained.clip(-1, 1)
    cost = lambda u: np.sum(weights*(matrix@u-target)**2)+.01*(u@u)
    assert np.max(abs(action)) <= 1
    assert diagnostics['ls_projected_gradient_inf'] < 1e-7
    assert cost(action) < cost(clipped)-1e-3
    assert np.linalg.norm(matrix@action-target) > 1  # unreachable target remains visible


def test_unactuated_directions_do_not_become_extra_motors():
    matrix = np.vstack([np.zeros((3, 6)), np.eye(6)])
    action, diagnostics = bounded_ls(matrix, np.zeros(9), np.r_[10., 10., 10., np.zeros(6)], .01)
    np.testing.assert_allclose(action, 0., atol=1e-12)
    assert diagnostics['weighted_acceleration_residual'] > 1


SAMPLES = Path(__file__).resolve().parents[1]/'validation'/'reference_samples'


def test_native_mpc_cannot_advance_or_modify_the_live_environment():
    import torch
    import mujoco
    from physical_fj.mechanics import PhysicalEnvironment
    from mujoco_reference.controller import NativeController
    from mujoco_reference.common import integration_state
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root/'prechecks/pure_cpht/feedback_precheck_model.json').read_text())
    options = dict(saved['settings']['pure'], candidates=2, elites=1, iterations=1, horizon_steps=2)
    env = PhysicalEnvironment(saved['control'], saved['settings'], 100000, clock='fixed', purpose='evaluation')
    try:
        controller = NativeController(env, saved, 'mujoco_mpc', options, 100000)
        p = env.base._env.physics
        _, before = integration_state(p.model.ptr, p.data.ptr)
        model_before = np.empty(mujoco.mj_sizeModel(p.model.ptr), dtype=np.uint8)
        model_after = np.empty_like(model_before)
        mujoco.mj_saveModel(p.model.ptr, buffer=model_before)
        raw_before = env.current['raw'].copy()
        action = controller.act(env.current, env.g, env.hold(), env.base.cur_t)
        _, after = integration_state(p.model.ptr, p.data.ptr)
        mujoco.mj_saveModel(p.model.ptr, buffer=model_after)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(model_before, model_after)
        np.testing.assert_array_equal(raw_before, env.current['raw'])
        assert np.max(abs(action)) <= 1
        assert controller.plans == 1
        assert controller.native_steps >= 20
        assert env.path_task.elapsed == 0
    finally:
        env.close()


@pytest.mark.parametrize('clock', ['fixed', 'irregular'])
@pytest.mark.parametrize('case', ['F', 'J'])
def test_instrumented_terrain_transition_and_rejection_of_corruption(tmp_path, clock, case):
    from mujoco_reference.transition import validate
    source = SAMPLES/clock/f'{case}_transition.json'
    if not source.exists():
        pytest.skip('Generate reference transition samples first')
    result = validate(source)
    assert result['passed']
    if clock == 'fixed':
        assert result['physics_substeps'] == 5
    sample = json.loads(source.read_text())
    # Reuse the model by absolute path only in the temporary test JSON.
    sample['replay_audit_only']['model_file'] = str(source.parent/sample['replay_audit_only']['model_file'])
    bad = copy.deepcopy(sample)
    bad['received_transition']['raw'][10] += 1.
    path = tmp_path/'bad_observation.json'
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        validate(path)
    bad = copy.deepcopy(sample)
    bad['received_transition']['actual_hold_seconds'] *= 2
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        validate(path)
    bad = copy.deepcopy(sample)
    bad['decision']['action_normalized'][0] = 1.1
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        validate(path)
