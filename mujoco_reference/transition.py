"""A real, instrumented held-action transition and a portable backend replay.

Truth and full integration state are audit fields, never policy inputs.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
import mujoco
import torch
from physical_fj.mechanics import full_mass
from pure_cpht.controller import Controller
from .common import copied, integration_state, step_native, write


def raw_schema(model, dimension):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(9)]
    items = []
    for block, unit in [('q', None), ('v', None), ('constraint_port', None)]:
        for j, name in enumerate(names):
            units = (['m']*2+['rad']*7 if block == 'q' else
                     ['m/s']*2+['rad/s']*7 if block == 'v' else ['N']*2+['N m']*7)
            items.append(dict(index=len(items), name=block+'_'+name, unit=units[j]))
    for name in ['remaining_COM_distance/path_length', 'remaining_time/horizon',
                 'target_speed/10_m_per_s', 'next_hold/reference_time']:
        items.append(dict(index=len(items), name=name, unit='1'))
    if dimension > 31:
        from environment_belief.config import FEATURES
        for name in FEATURES:
            items.append(dict(index=len(items), name=name, unit='see environment_belief/observer.py'))
    if len(items) != dimension:
        raise ValueError('Unknown observation schema')
    return items


@torch.no_grad()
def predict_learned(saved, before, action, duration, physics_dt, previous_action):
    ctrl = Controller(saved['pure_model'], saved['settings']['pure'], saved['neutral'],
                      gait=saved['gait'], feedback_only=True)
    ctrl.previous = ctrl.robot.t(previous_action)
    ctrl.act(before, float(before['raw'][29])*10., float(before['raw'][30])*saved['control']['reference_time'], 0.)
    r = ctrl.robot
    q, v = r.t(before['raw'][:9])[None], r.t(before['raw'][9:18])[None]
    scan, u = r.t(before['scan']), r.t(action)[None]
    count = max(1, round(duration/physics_dt))
    dt = duration/count
    first = None
    for k in range(count):
        d = r.quantities(q, v)
        contact = ctrl.contact.solve(q, v, u, scan, d, ctrl.force_correction, k*dt)
        acceleration = torch.linalg.solve(d['M'], (u@r.B.T+contact['port']-d['c'])[:, :, None]).squeeze(-1)
        if first is None:
            first = dict(mass=d['M'][0].numpy(), drift=d['c'][0].numpy(),
                         constraint_port=contact['port'][0].numpy(), acceleration=acceleration[0].numpy(),
                         joint_damping=r.damping.numpy())
        v = v+dt*acceleration
        q = q+dt*v
    return dict(q=q[0].numpy(), v=v[0].numpy(), integrator='semi-implicit Euler of learned force model',
                dt=dt, substeps=count, model_parameter_refits=0,
                information='current raw state, estimated scan/friction, previous action, current force port',
                first_step_model=first)


def capture(env, controller, saved, path):
    """Execute exactly one selected action; snapshot methods never touch live data."""
    path = Path(path)
    p = env.base._env.physics
    model, live = p.model.ptr, p.data.ptr
    before = copy.deepcopy(env.current)
    time_before, requested = float(env.base.cur_t), float(env.hold())
    previous_action = live.ctrl.copy()
    # Select control before recording the integration state, and explicitly
    # verify that planning cannot change the live integration state.
    signature, prior = integration_state(model, live)
    action = np.asarray(controller.act(before, env.g, requested, time_before), dtype=np.float32)
    _, state = integration_state(model, live)
    if not np.array_equal(prior, state):
        raise AssertionError('Control selection mutated live MuJoCo integration state')
    scratch = copied(model, live)
    substeps, pending = [], {}
    original_force, original_after = env.base._point_force, env.base._after_physics_step

    def force():
        result = original_force()
        d = copied(model, live)
        # The wrapper calls _point_force before dm_control.before_step installs
        # the newly issued control. Audit the control that this substep executes.
        d.ctrl[:] = action
        mujoco.mj_forward(model, d)
        mass = full_mass(model, d)
        external = np.zeros(model.nv)
        for body in range(1, model.nbody):
            force_torque = d.xfrc_applied[body]
            if np.any(force_torque):
                mujoco.mj_applyFT(model, d, force_torque[:3], force_torque[3:], d.xipos[body], body, external)
        residual = mass@d.qacc+d.qfrc_bias-(d.qfrc_actuator+d.qfrc_passive+d.qfrc_applied+external+d.qfrc_constraint)
        contact_records = []
        for i in range(d.ncon):
            contact = d.contact[i]
            force_coordinates = np.empty(6)
            mujoco.mj_contactForce(model, d, i, force_coordinates)
            contact_records.append(dict(geom1=int(contact.geom1), geom2=int(contact.geom2),
                                        distance=float(contact.dist), sliding_friction=float(contact.friction[0]),
                                        force_contact_coordinates=force_coordinates))
        pending.clear()
        pending.update(time_backend=float(d.time), q=d.qpos.copy(), v=d.qvel.copy(), acceleration=d.qacc.copy(),
                       mass=mass, bias=d.qfrc_bias.copy(), actuator_force=d.qfrc_actuator.copy(),
                       passive_force=d.qfrc_passive.copy(), applied_force=d.qfrc_applied.copy()+external,
                       constraint_force=d.qfrc_constraint.copy(), contacts=d.ncon, contact_records=contact_records,
                       force_balance_inf=float(np.max(np.abs(residual))))
        return result

    def after(dt):
        original_after(dt)
        mass = pending['mass']
        damping = np.diag(model.dof_damping)
        if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP):
            damping = np.zeros_like(damping)
        expected_v = pending['v']+dt*np.linalg.solve(mass+dt*damping, mass@pending['acceleration'])
        expected_q = pending['q']+dt*expected_v
        row = dict(pending, dt=float(dt), q_next=live.qpos.copy(), v_next=live.qvel.copy(),
                   euler_velocity_defect_inf=float(np.max(abs(expected_v-live.qvel))),
                   euler_position_defect_inf=float(np.max(abs(expected_q-live.qpos))))
        substeps.append(row)

    env.base._point_force, env.base._after_physics_step = force, after
    try:
        successor, info, _ = env.step(action)
    finally:
        env.base._point_force, env.base._after_physics_step = original_force, original_after
    # Contact runs have no wind. A nonzero force/intervention requires recording
    # its causal substep inputs as well; do not silently claim an exact replay.
    if env.config['experiment'] == 'wind':
        raise ValueError('Use this audit for the terrain/contact experiment')
    replay = copied(model, scratch)
    replay.ctrl[:] = action
    mujoco.mj_step1(model, replay)
    for row in substeps:
        replay.qfrc_applied[:] = row['applied_force']
        replay.xfrc_applied[:] = 0.
        step_native(model, replay, bool(p.legacy_step))
    learned = predict_learned(saved, before, action, info['h'], model.opt.timestep, previous_action)
    model_file = path.with_suffix('.mjb')
    model_file.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(model_file))
    observed_q, observed_v = live.qpos.copy(), live.qvel.copy()
    actual_first, learned_first = substeps[0], learned['first_step_model']
    true_drift = actual_first['bias']-actual_first['passive_force']
    def relative(a, b):
        return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-8))
    velocity_error = np.asarray(learned['v'])-observed_v
    local = dict(mass_relative_frobenius=relative(learned_first['mass'], actual_first['mass']),
                 drift_relative_l2=relative(learned_first['drift'], true_drift),
                 contact_port_relative_l2=relative(learned_first['constraint_port'], actual_first['constraint_force']),
                 acceleration_relative_l2=relative(learned_first['acceleration'], actual_first['acceleration']),
                 interpretation='one local operating point; force/drift norms combine translational and rotational coordinates')
    from fjbench.contact import query
    torso = p.model.name2id('torso', 'body')
    local_truth = query(env.contact_map, [float(scratch.xpos[torso, 0])])
    # Separate environment-estimation effects from the calibrated contact
    # closure at the SAME state/action. This is an explicitly privileged local
    # diagnostic, never a deployed policy or a training target from the future.
    oracle_packet = copy.deepcopy(before)
    scan_truth = query(env.contact_map, np.asarray(before['scan'])[:, 0])
    oracle_packet['scan'] = np.column_stack((np.asarray(before['scan'])[:, 0], scan_truth['height'], scan_truth['slope']))
    oracle_packet['environment'] = dict(oracle_packet.get('environment', {}), friction=float(local_truth['mu'][0]))
    oracle_packet['scan_source'] = 'audit-only current terrain truth'
    oracle_prediction = predict_learned(saved, oracle_packet, action, info['h'], model.opt.timestep, previous_action)
    oracle_prediction['access'] = 'audit-only exact current terrain samples and root-location friction; model weights unchanged'
    oracle_velocity_error = np.asarray(oracle_prediction['v'])-observed_v
    sample = dict(schema='cheetah_transition_v1', origin='new replay of supplied checkpoint; not extracted from old video',
        checkpoint_sha256=saved['_sha256'], case=saved['settings']['case'],
        pipeline_input=dict(time_seconds=time_before, raw=before['raw'], raw_schema=raw_schema(model, len(before['raw'])),
                            raw_dtype='float32', scan=before['scan'], scan_columns=['x_m', 'height_m', 'slope'],
                            scan_source=before.get('scan_source'), environment=before.get('environment'),
                            information_contract=before.get('information_contract'), previous_action=previous_action,
                            requested_hold_seconds=requested, reference_time_seconds=env.config['reference_time'], target_speed_m_per_s=env.g,
                            state_backend_q=scratch.qpos.copy(), state_backend_v=scratch.qvel.copy()),
        decision=dict(method=getattr(controller, 'method', 'learned_feedback'), action_normalized=action,
                      actuator_gear=model.actuator_gear[:, 0], joint_motor_torque=model.actuator_gear[:, 0]*action,
                      diagnostics=controller.diagnostics),
        received_transition=dict(time_seconds=float(env.base.cur_t), raw=successor['raw'],
                                 q=observed_q, v=observed_v, actual_hold_seconds=info['h'],
                                 physics_substeps=info['physics_steps'], info=info),
        predictions=dict(mujoco=dict(q=replay.qpos.copy(), v=replay.qvel.copy()), learned_pH=learned,
                         learned_pH_truth_environment=oracle_prediction,
                         instantaneous_model_diagnostics=local),
        checks=dict(planning_left_live_state_unchanged=True,
                    replay_q_max_abs=float(np.max(abs(replay.qpos-observed_q))),
                    replay_v_max_abs=float(np.max(abs(replay.qvel-observed_v))),
                    learned_q_l2=float(np.linalg.norm(np.asarray(learned['q'])-observed_q)),
                    learned_v_l2=float(np.linalg.norm(np.asarray(learned['v'])-observed_v)),
                    learned_v_translational_l2_m_per_s=float(np.linalg.norm(velocity_error[:2])),
                    learned_v_angular_l2_rad_per_s=float(np.linalg.norm(velocity_error[2:])),
                    truth_environment_learned_v_translational_l2_m_per_s=float(np.linalg.norm(oracle_velocity_error[:2])),
                    truth_environment_learned_v_angular_l2_rad_per_s=float(np.linalg.norm(oracle_velocity_error[2:])),
                    force_balance_inf=max(r['force_balance_inf'] for r in substeps),
                    euler_velocity_defect_inf=max(r['euler_velocity_defect_inf'] for r in substeps),
                    euler_position_defect_inf=max(r['euler_position_defect_inf'] for r in substeps)),
        substep_audit=substeps,
        replay_audit_only=dict(mujoco_version=mujoco.__version__, dm_control_legacy_step=bool(p.legacy_step),
                               integrator=int(model.opt.integrator), model_file=model_file.name,
                               model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                               integration_signature=signature, integration_state=state,
                               backend_time_before=float(scratch.time), terrain=env.audit_truth(), local_terrain_truth=local_truth,
                               access='compiled full model and integration state; never an ordinary policy input'))
    write(path, sample)
    return sample


def validate(path, tolerance=1e-7):
    """Standalone validation: raw contract, force equation, held-step replay."""
    import json
    path = Path(path)
    row = json.loads(path.read_text())
    if row['schema'] != 'cheetah_transition_v1':
        raise ValueError('Unknown transition schema')
    audit, received = row['replay_audit_only'], row['received_transition']
    if audit['mujoco_version'] != mujoco.__version__:
        raise RuntimeError('Binary replay requires the recorded MuJoCo version '+audit['mujoco_version'])
    model_path = path.parent/audit['model_file']
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != audit['model_sha256']:
        raise AssertionError('Backend model snapshot does not match its recorded hash')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    d = mujoco.MjData(model)
    mujoco.mj_setState(model, d, np.asarray(audit['integration_state']), audit['integration_signature'])
    action = np.asarray(row['decision']['action_normalized'])
    raw = np.asarray(row['pipeline_input']['raw'])
    if len(raw) not in (31, 60) or raw.shape != (len(row['pipeline_input']['raw_schema']),) or not np.isfinite(raw).all():
        raise AssertionError('Invalid raw observation')
    if action.shape != (6,) or not np.isfinite(action).all() or np.any(abs(action) > 1.+1e-7):
        raise AssertionError('Invalid action')
    np.testing.assert_allclose(raw[:9], d.qpos, rtol=0., atol=2e-6)
    np.testing.assert_allclose(raw[9:18], d.qvel, rtol=0., atol=2e-6)
    elapsed = received['actual_hold_seconds']
    expected_steps = received['physics_substeps']
    if expected_steps < 1 or elapsed <= 0:
        raise AssertionError('A received transition must execute a positive number of physics substeps')
    if len(row['substep_audit']) != expected_steps:
        raise AssertionError('Substep count does not match received transition')
    if abs(elapsed-expected_steps*model.opt.timestep) > 1e-9:
        raise AssertionError('Physical time does not match composed substeps')
    if abs(received['time_seconds']-row['pipeline_input']['time_seconds']-elapsed) > 1e-9:
        raise AssertionError('Inconsistent transition timestamps')
    if elapsed > row['pipeline_input']['requested_hold_seconds']+1e-9:
        raise AssertionError('Executed hold exceeds the disclosed requested hold')
    if abs(raw[30]*row['pipeline_input']['reference_time_seconds']-row['pipeline_input']['requested_hold_seconds']) > 2e-8:
        raise AssertionError('Requested hold disagrees with its raw observation feature')
    next_raw = np.asarray(received['raw'])
    if next_raw.shape != raw.shape or not np.isfinite(next_raw).all():
        raise AssertionError('Invalid received observation')
    np.testing.assert_allclose(next_raw[:9], received['q'], rtol=0., atol=2e-6)
    np.testing.assert_allclose(next_raw[9:18], received['v'], rtol=0., atol=2e-6)
    np.testing.assert_allclose(row['decision']['joint_motor_torque'], model.actuator_gear[:, 0]*action,
                               rtol=0., atol=1e-8)
    d.ctrl[:] = action
    mujoco.mj_step1(model, d)
    max_force_defect = 0.
    for sub in row['substep_audit']:
        np.testing.assert_allclose(d.qpos, sub['q'], rtol=0., atol=tolerance)
        np.testing.assert_allclose(d.qvel, sub['v'], rtol=0., atol=tolerance)
        equation = (np.asarray(sub['mass'])@sub['acceleration']+sub['bias']
                    -np.asarray(sub['actuator_force'])-sub['passive_force']-sub['applied_force']-sub['constraint_force'])
        max_force_defect = max(max_force_defect, float(np.max(abs(equation))))
        np.testing.assert_allclose(sub['actuator_force'][3:], model.actuator_gear[:, 0]*action, rtol=0., atol=1e-8)
        if max(sub['euler_velocity_defect_inf'], sub['euler_position_defect_inf']) > tolerance:
            raise AssertionError('Substep does not match the recorded implicit-damping Euler update')
        d.qfrc_applied[:] = sub['applied_force']
        d.xfrc_applied[:] = 0.
        step_native(model, d, audit['dm_control_legacy_step'])
        np.testing.assert_allclose(d.qpos, sub['q_next'], rtol=0., atol=tolerance)
        np.testing.assert_allclose(d.qvel, sub['v_next'], rtol=0., atol=tolerance)
    q_error = float(np.max(abs(d.qpos-received['q'])))
    v_error = float(np.max(abs(d.qvel-received['v'])))
    if max(q_error, v_error, max_force_defect) > tolerance:
        raise AssertionError(f'Transition fails replay/force tolerance: q={q_error}, v={v_error}, force={max_force_defect}')
    result = dict(passed=True, tolerance=tolerance, replay_q_max_abs=q_error, replay_v_max_abs=v_error,
                  force_balance_inf=max_force_defect, actual_hold_seconds=elapsed, physics_substeps=expected_steps)
    return result
