"""Simulation sensor boundary and optional hidden friction interventions.

Only this boundary reads the terrain map. The observer receives noisy geometric
frames and the declared 31-channel state. Friction truth is audit-only.
"""
from collections import deque
import numpy as np
from .observer import Observer
from .config import contract


class Runtime:
    def __init__(self, env, nominal, settings, seed):
        from pure_cpht.mechanics import public_geometry
        self.env, self.o = env, settings['sensing']
        self.contract = contract(settings)
        self.observer = Observer(public_geometry(nominal), env.config['contact_offsets'], self.o)
        self.rng = np.random.default_rng(seed+12347)
        self.queue = deque()
        self.frames_attempted = self.frames_delivered = self.frames_dropped = 0
        self.last_decision = None
        self.next_sample = 0.
        self.scale = 1.
        self.next_event = 0
        self.events_applied = []
        event_rng = np.random.default_rng(seed+20117)
        self.events = sorted([dict(time=max(0., event['time']+event_rng.uniform(-self.o['friction_event_jitter'], self.o['friction_event_jitter'])),
            scale=event['scale']*np.exp(event_rng.normal(0., self.o['friction_scale_jitter']))) for event in self.o['friction_events']],
            key=lambda event: event['time'])
        m = env.base._env.physics.model
        self.ground = [i for i in range(m.ngeom) if int(m.geom_bodyid[i]) == 0]
        self.ground_friction = m.geom_friction[self.ground].copy()
        original_force = env.base._point_force
        def point_force():
            self.apply_events(float(env.base.cur_t+getattr(env.base, '_micro_elapsed', 0.)))
            return original_force()
        env.base._point_force = point_force
        original_after = env.base._after_physics_step
        def after_step(dt):
            original_after(dt)
            self.sample_due(float(env.base.cur_t+getattr(env.base, '_micro_elapsed', 0.)+dt))
        env.base._after_physics_step = after_step
        self.apply_events(0.)
        self.sample_due(0.)

    def apply_events(self, time):
        events = self.events
        while self.next_event < len(events) and events[self.next_event]['time'] <= time+1e-9:
            event = events[self.next_event]
            self.scale = event['scale']
            # Absolute scale relative to the sampled course, never cumulative.
            self.env.base._env.physics.model.geom_friction[self.ground] = self.ground_friction*self.scale
            self.events_applied.append(dict(requested_time=event['time'], applied_time=float(time), scale=self.scale))
            self.next_event += 1

    def sample_due(self, time):
        if self.o['mode'] != 'sensed_belief':
            return
        if time+1e-9 < self.next_sample:
            return
        # Config validation keeps the sensor period at least one physics step.
        while self.next_sample <= time+1e-9:
            self.next_sample += self.o['sensor_period']
        self.frames_attempted += 1
        if self.rng.random() < self.o['sensor_dropout']:
            self.frames_dropped += 1
            return
        from fjbench.contact import query
        p = self.env.base._env.physics
        torso = p.model.name2id('torso', 'body')
        # Root translations and public torso offset give the same exact pose
        # already present in the original instrumented state contract.
        root = p.data.qpos[:2]+p.model.body_pos[torso, [0, 2]]
        x = root[0]+np.asarray(self.env.config['contact_offsets'])
        truth = query(self.env.contact_map, x)
        frame = dict(time=float(time), x=x.copy(),
            height=truth['height']+self.rng.normal(0., self.o['height_std'], len(x)),
            slope=truth['slope']+self.rng.normal(0., self.o['slope_std'], len(x)))
        self.queue.append((time+self.o['sensor_delay'], frame))

    def observe(self, raw):
        time = float(self.env.base.cur_t)
        torso = self.env.base._env.physics.model.name2id('torso', 'body')
        root = np.asarray(raw[:2], dtype=float)+self.env.base._env.physics.model.body_pos[torso, [0, 2]]
        if self.last_decision is None or time > self.last_decision+1e-9:
            self.observer.predict(time, root[0])
            while self.queue and self.queue[0][0] <= time+1e-9:
                _, frame = self.queue.popleft()
                self.observer.assimilate_geometry(frame)
                self.frames_delivered += 1
            self.observer.assimilate_state(raw)
            self.last_decision = time
        scan, features, details = self.observer.packet(root[1])
        vector = np.asarray(raw[:31], dtype=np.float32)
        if self.o['mode'] != 'state':
            vector = np.r_[vector, features].astype(np.float32)
        return dict(raw=vector, scan=scan, environment=details,
            information_contract=self.contract, scan_source='causal_environment_observer')

    def audit(self):
        """Evaluation diagnostics. Never called by a policy or the observer."""
        from fjbench.contact import query
        p = self.env.base._env.physics
        torso = p.model.name2id('torso', 'body')
        x = float(p.data.qpos[0]+p.model.body_pos[torso, 0])
        truth = query(self.env.contact_map, [x])
        side = query(self.env.contact_map, [x-.05, x+.05])
        slope_difference=float(side['slope'][1]-side['slope'][0])
        curvature=slope_difference/.1/(1+float(truth['slope'][0])**2)**1.5
        _, _, estimate = self.observer.packet(float(p.data.qpos[1]+p.model.body_pos[torso, 2]))
        return dict(time=float(self.env.base.cur_t), root_x=x, estimate=estimate,
            truth=dict(height=float(truth['height'][0]), slope=float(truth['slope'][0]),
                curvature=curvature,curvature_reference_valid=abs(slope_difference)<.4,
                curvature_reference='slope finite difference over +/-0.05m; excludes large slope jumps',
                local_patch_friction=float(truth['mu'][0])*self.scale*self.env._friction),
            contact_truth=[{key:record[key] for key in ('body','normal_force','slip_speed','mu_effective')}
                for record in self.env.contact_audit.records] if self.env.contact_audit else [],
            sensor=dict(attempted=self.frames_attempted, delivered=self.frames_delivered,
                dropped=self.frames_dropped, pending=len(self.queue)),
            friction_events=list(self.events_applied), audit_only=True)


def summarize(rows):
    if not rows:
        return {}
    result = dict(samples=len(rows), sensor=rows[-1]['sensor'], friction_events=rows[-1]['friction_events'])
    for name in ('height', 'slope'):
        errors = [r['estimate'][name]-r['truth'][name] for r in rows if r['estimate']['geometry_valid']]
        result[name+'_rmse'] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    informative = [r for r in rows if r['estimate']['sliding_evidence']]
    result['informative_sliding_samples'] = len(informative)
    # A root patch is a context reference; front/back contacts can occupy other
    # patches. Contact-pair friction is recorded separately in contact_audit.
    result['friction_reference'] = 'terrain patch below torso; contact-pair truth remains in contact_audit'
    result['friction_reference_rmse_on_sliding'] = float(np.sqrt(np.mean([
        (r['estimate']['friction']-r['truth']['local_patch_friction'])**2 for r in informative]))) if informative else None
    result['friction_updates'] = rows[-1]['estimate']['friction_updates']
    curves=[r['estimate']['curvature']-r['truth']['curvature'] for r in rows
        if r['estimate']['curvature_valid'] and r['truth']['curvature_reference_valid']]
    result['curvature_finite_difference_rmse']=float(np.sqrt(np.mean(np.square(curves)))) if curves else None
    pairs=[(r['estimate']['friction']-contact['mu_effective'],contact['normal_force']) for r in rows
        for contact in r['contact_truth'] if contact['normal_force']>5. and contact['slip_speed']>.1]
    result['loaded_sliding_contact_pairs']=len(pairs)
    result['friction_pair_rmse_on_audit_sliding']=float(np.sqrt(np.average(
        [error**2 for error,_ in pairs],weights=[weight for _,weight in pairs]))) if pairs else None
    return result
