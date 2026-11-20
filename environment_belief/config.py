"""Versioned information contracts. Legacy runs omit this configuration."""
import math

MODES = ('state', 'state_belief', 'sensed_belief')
FEATURES = tuple([f'height_{i}' for i in range(7)] + [f'slope_{i}' for i in range(7)] + [
    'curvature', 'friction', 'height_std', 'slope_std', 'curvature_std', 'friction_std',
    'geometry_age_s', 'friction_age_s', 'geometry_valid', 'friction_valid', 'sliding_evidence', 'normal_load_100N',
    'geometry_sensor_age_s', 'geometry_sensor_valid', 'curvature_valid'])
DEFAULTS = dict(mode='state', sensor_period=.02, sensor_delay=0., sensor_dropout=0.,
    height_std=.01, slope_std=.03, geometry_process=.005, friction_process=.08,
    friction_prior=.6, friction_log_std=.65, friction_cell=.75, friction_measurement_std=.25,
    slip_threshold=.1, minimum_load=5., inverse_residual_limit=.35, contact_band=.035,
    impact_dwell=.02, age_cap=5., friction_event_jitter=0., friction_scale_jitter=0., friction_events=[])


def options(value):
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown sensing options: {sorted(unknown)}')
    out = dict(DEFAULTS, **value)
    if out['mode'] not in MODES:
        raise ValueError(f'Sensing mode must be one of {MODES}')
    for key, val in out.items():
        if key in ('mode', 'friction_events'):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f'Finite numeric sensing option required: {key}')
        if key in ('sensor_delay', 'sensor_dropout', 'impact_dwell', 'friction_event_jitter', 'friction_scale_jitter'):
            if val < 0:
                raise ValueError(f'Nonnegative sensing option required: {key}')
        elif val <= 0:
            raise ValueError(f'Positive sensing option required: {key}')
    if out['sensor_dropout'] > 1:
        raise ValueError('Sensor dropout must be in [0,1]')
    previous = -1.
    events = []
    for event in out['friction_events']:
        if set(event) != {'time', 'scale'}:
            raise ValueError('Each friction event needs time and scale')
        time, scale = event['time'], event['scale']
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (time, scale)):
            raise ValueError('Finite friction event time and scale required')
        if time < 0 or time <= previous or scale <= 0:
            raise ValueError('Friction events need increasing nonnegative times and positive absolute scales')
        previous = time
        events.append(dict(time=float(time), scale=float(scale)))
    out['friction_events'] = events
    return out


def contract(s):
    if 'sensing' not in s:
        return dict(mode='legacy', schema='legacy', augmented=False)
    o = s['sensing']
    return dict(schema='environment_belief_v1', mode=o['mode'],
        augmented=o['mode'] != 'state', base_dimension=31,
        observation_dimension=31 + (len(FEATURES) if o['mode'] != 'state' else 0),
        extra_features=list(FEATURES) if o['mode'] != 'state' else [],
        external_geometry_sensor=o['mode'] == 'sensed_belief', direct_material_sensor=False,
        state_sensor='exact q, v, aggregate constraint-force port at control decisions',
        inference='known robot geometry; causal original-state history; optional delayed noisy geometry rays',
        estimator='fixed-parameter local quadratic Kalman filter and gated contact-force inverse',
        original_state_history_available_to_both=True,
        calibration_labels='shared q/v/acceleration/aggregate force labels; new presets fit contact ports without per-contact labels',
        contact_supervision=s.get('pure',{}).get('contact_supervision','point_labels'),
        test_parameter_refitting=False)
