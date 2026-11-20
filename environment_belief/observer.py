"""Causal, approximate terrain/friction filter with no simulator or map access.

The force inverse uses unconstrained least squares and a load/residual gate.
Its result never depends on the current friction cone. Multiple simultaneous
contacts can remain ambiguous; reported variance is a heuristic, not a proved
posterior calibration guarantee.
"""
import math
import numpy as np
from .config import FEATURES


def rotate(angle, point):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c*point[0]+s*point[1], -s*point[0]+c*point[1]])


def point_kinematics(geometry, q):
    origins = [np.zeros(2)]
    angles = [0.]
    hinges = np.zeros((9, 2))
    ancestors = np.zeros((8, 9))
    for b in range(1, 8):
        parent = geometry['parents'][b]
        origin = origins[parent]+rotate(angles[parent], geometry['positions'][b])
        ancestors[b] = ancestors[parent]
        if b == 1:
            origin = origin+q[:2]
            angle, joint = q[2], 2
        else:
            joint = geometry['joint'][b]
            angle = angles[parent]+q[joint]
        hinges[joint] = origin
        ancestors[b, joint] = 1.
        origins.append(origin)
        angles.append(angle)
    points, jacobians, radii, bodies = [], [], [], []
    for item in geometry['points']:
        b = item['body']
        p = origins[b]+rotate(angles[b], item['local'])
        delta = p-hinges
        J = np.stack((delta[:, 1], -delta[:, 0]))*ancestors[b]
        J[0, 0], J[1, 1] = 1., 1.
        points.append(p)
        jacobians.append(J)
        radii.append(item['radius'])
        bodies.append(b)
    return np.asarray(points), np.asarray(jacobians), np.asarray(radii), np.asarray(bodies), ancestors


class Observer:
    def __init__(self, geometry, offsets, options):
        if len(offsets) != 7:
            raise ValueError('The environment-belief v1 schema uses seven geometric probe locations')
        self.geometry, self.offsets, self.o = geometry, np.asarray(offsets), options
        self.center = 0.
        self.a = np.zeros(3)
        self.P = np.diag([.25, .25, .25])
        self.log_mu = math.log(options['friction_prior'])
        self.mu_variance = options['friction_log_std']**2
        self.time = None
        self.geometry_time = self.friction_time = None
        self.external_time = None
        self.mu_anchor = None
        self.contact_since = None
        self.sliding = False
        self.load = 0.
        self.inverse_error = None
        self.external_updates = self.contact_updates = self.friction_updates = 0
        self.last_scan = None
        self.curvature_valid = False

    def predict(self, time, center):
        if self.time is not None and time < self.time-1e-9:
            raise ValueError('Observer timestamps must be causal and monotone')
        dt = 0. if self.time is None else max(0., time-self.time)
        dx = center-self.center
        T = np.array([[1., dx, .5*dx*dx], [0., 1., dx], [0., 0., 1.]])
        self.a = T@self.a
        self.P = T@self.P@T.T + self.o['geometry_process']*dt*np.diag([1., 1., 2.])
        self.mu_variance = min(2., self.mu_variance+self.o['friction_process']*dt)
        if self.mu_anchor is not None and abs(center-self.mu_anchor) > self.o['friction_cell']:
            self.mu_variance = max(self.mu_variance, self.o['friction_log_std']**2)
            self.friction_time = None
            self.mu_anchor = center
        self.center, self.time = float(center), float(time)

    def geometry_update(self, x, height, slope=None, timestamp=None):
        x, height = np.asarray(x), np.asarray(height)
        dx = x-self.center
        H = np.column_stack((np.ones_like(dx), dx, .5*dx*dx))
        values = height
        variances = np.full(len(x), self.o['height_std']**2)
        if slope is not None:
            H = np.vstack((H, np.column_stack((np.zeros_like(dx), np.ones_like(dx), dx))))
            values = np.r_[values, slope]
            variances = np.r_[variances, np.full(len(x), self.o['slope_std']**2)]
        # Favour the local patch when distant terrain has a different shape.
        variances *= np.tile(1.+(dx/1.25)**4, 2 if slope is not None else 1)
        S = H@self.P@H.T+np.diag(variances)
        K = np.linalg.solve(S, H@self.P).T
        self.a += K@(values-H@self.a)
        I_KH = np.eye(3)-K@H
        self.P = I_KH@self.P@I_KH.T+(K*variances)@K.T
        self.P = .5*(self.P+self.P.T)
        self.geometry_time = float(self.time if timestamp is None else timestamp)
        if slope is not None:
            self.last_scan = np.column_stack((x, height, slope))
            residual = np.sqrt(np.mean((H@self.a-values)**2))
            self.curvature_valid = bool(residual < .15)
            self.external_time = float(self.time if timestamp is None else timestamp)
            self.external_updates += 1
        else:
            self.contact_updates += 1

    def assimilate_geometry(self, frame):
        # A delayed frame retains its acquisition timestamp and world frame.
        self.geometry_update(frame['x'], frame['height'], frame['slope'], frame['time'])

    def friction_update(self, normal, tangent, speed, residual, time):
        self.sliding = False
        if normal < self.o['minimum_load'] or residual > self.o['inverse_residual_limit']:
            self.contact_since = None
            return
        if self.contact_since is None:
            self.contact_since = time
        if abs(speed) < self.o['slip_threshold'] or time-self.contact_since < self.o['impact_dwell']:
            # A sticking contact supplies an inequality. Retain broad uncertainty.
            return
        ratio = abs(tangent)/max(normal, 1e-8)
        if not .03 <= ratio <= 2.5 or tangent*speed >= 0:
            return
        z = math.log(ratio)
        R = self.o['friction_measurement_std']**2+residual**2
        gain = self.mu_variance/(self.mu_variance+R)
        # Bounded innovations avoid a single impact dominating the estimate.
        self.log_mu += gain*np.clip(z-self.log_mu, -1., 1.)
        self.mu_variance = max(.01, (1-gain)*self.mu_variance)
        self.friction_time, self.mu_anchor = float(time), self.center
        self.sliding = True
        self.friction_updates += 1

    def assimilate_state(self, raw):
        q, v, port = np.asarray(raw[:9]), np.asarray(raw[9:18]), np.asarray(raw[18:27])
        p, J, radius, bodies, ancestors = point_kinematics(self.geometry, q)
        slope = float(np.clip(self.a[1], -1.5, 1.5))
        normal = np.array([-slope, 1.])/math.sqrt(1+slope*slope)
        tangent = np.array([normal[1], -normal[0]])
        points = p-radius[:, None]*normal
        offset = -radius[:, None]*normal
        J += np.stack((offset[:, 1], -offset[:, 0]), 1)[:, :, None]*ancestors[bodies, None, :]
        # Infer a small contact set from public geometry and the current state.
        # This is approximate and receives no true contact IDs or surface labels.
        order = np.argsort(points[:, 1])
        selected = []
        for i in order:
            if points[i, 1] > points[order[0], 1]+self.o['contact_band']:
                break
            if bodies[i] not in [bodies[j] for j in selected]:
                selected.append(i)
            if len(selected) == 2:
                break
        self.load = max(0., float(port[1]))
        self.sliding = False
        if not selected or self.load < self.o['minimum_load']:
            self.contact_since = None
            return
        A = np.concatenate([J[i].T for i in selected], axis=1)
        # Joint-stop forces do not act directly on the three root coordinates.
        weights = np.ones(9)
        limits = np.asarray(self.geometry['limits'])
        near_limit = (q[3:] < limits[:, 0]+.03) | (q[3:] > limits[:, 1]-.03)
        weights[3:][near_limit] = 0.
        design = weights[:, None]*A
        forces = np.linalg.lstsq(np.vstack((design, .01*np.eye(A.shape[1]))),
                                 np.r_[weights*port, np.zeros(A.shape[1])], rcond=None)[0].reshape(-1, 2)
        residual = np.linalg.norm(weights*(A@forces.ravel()-port))/max(1., np.linalg.norm(weights*port))
        self.inverse_error = float(residual)
        fn = forces@normal
        loaded = fn > self.o['minimum_load']
        if residual > self.o['inverse_residual_limit'] or not loaded.any():
            self.contact_since = None
            return
        ids = np.asarray(selected)[loaded]
        # Loaded candidate points provide sparse elevation evidence.
        self.geometry_update(points[ids, 0], points[ids, 1])
        j = int(np.argmax(fn))
        velocity = J[selected[j]]@v
        self.friction_update(float(fn[j]), float(forces[j]@tangent), float(velocity@tangent), float(residual), self.time)

    def packet(self, root_z):
        x = self.center+self.offsets
        height = self.a[0]+self.a[1]*self.offsets+.5*self.a[2]*self.offsets**2
        slope = self.a[1]+self.a[2]*self.offsets
        if self.last_scan is not None:
            s = self.last_scan
            inside = (x >= s[0, 0]) & (x <= s[-1, 0])
            height[inside] = np.interp(x[inside], s[:, 0], s[:, 1])
            slope[inside] = np.interp(x[inside], s[:, 0], s[:, 2])
        scan = np.column_stack((x, height, np.clip(slope, -1.5, 1.5)))
        mu = float(np.clip(math.exp(self.log_mu), .03, 2.5))
        age = lambda time: self.o['age_cap'] if time is None else min(self.o['age_cap'], max(0., self.time-time))
        std = np.sqrt(np.maximum(0., np.diag(self.P)))
        mu_std = mu*math.sqrt(self.mu_variance)
        curvature = self.a[2]/(1+self.a[1]**2)**1.5
        # Delta-method uncertainty for the signed curvature.
        gradient = np.array([0., -3*self.a[1]*self.a[2]/(1+self.a[1]**2)**2.5, (1+self.a[1]**2)**-1.5])
        curvature_std = math.sqrt(max(0., gradient@self.P@gradient))
        extras = np.r_[height-root_z, scan[:, 2], curvature, mu, std[:2], curvature_std, mu_std,
                       age(self.geometry_time), age(self.friction_time),
                       float(self.geometry_time is not None), float(self.friction_time is not None),
                       float(self.sliding), self.load/100., age(self.external_time),
                       float(self.external_time is not None), float(self.curvature_valid)].astype(np.float32)
        if len(extras) != len(FEATURES) or not np.isfinite(extras).all():
            raise FloatingPointError('Invalid environment-belief feature vector')
        details = dict(friction=mu, friction_std=mu_std, height=float(self.a[0]), slope=float(self.a[1]),
            height_std=float(std[0]), slope_std=float(std[1]), curvature_std=curvature_std,
            curvature=float(curvature), curvature_valid=self.curvature_valid,
            geometry_age=age(self.geometry_time), friction_age=age(self.friction_time),
            geometry_sensor_age=age(self.external_time), geometry_sensor_valid=self.external_time is not None,
            geometry_valid=self.geometry_time is not None, friction_valid=self.friction_time is not None,
            sliding_evidence=self.sliding, inverse_relative_error=self.inverse_error,
            external_geometry_updates=self.external_updates, contact_geometry_updates=self.contact_updates,
            friction_updates=self.friction_updates)
        return scan, extras, details
