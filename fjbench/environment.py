"""Measured Cheetah cases with explicit contact environments. Hidden parameters stay in the backend."""
import hashlib
import importlib.resources
import math
import types
import numpy as np
from bench.env import make_common_env
from .config import observation_dim,is_path_task


def ou_step(value,kappa,sigma,dt,rng):
    if kappa<=0 or dt<=0 or sigma<0:
        raise ValueError("OU parameters require kappa>0, dt>0, sigma>=0")
    decay=math.exp(-kappa*dt)
    return decay*value+sigma*math.sqrt(-math.expm1(-2*kappa*dt)/(2*kappa))*rng.normal()


class Environment:
    def __init__(self,config,seed,episode=0,cohort="iid",clock=None,progress=1.,difficulty=None,
                 environment_spec=None,start_x=None,purpose="train",initial_speed=None):
        import mujoco
        self.config=config
        self.sensor_rng=np.random.default_rng(seed+51)
        self.wind_rng=np.random.default_rng(seed+67)
        self.environment_sensor_rng=np.random.default_rng(seed+79)
        self.odometry_rng=np.random.default_rng(seed+83)
        rng=np.random.default_rng(seed+31)
        c=config
        is_contact=c["experiment"]=="contact"
        path=is_path_task(c);self.path_task=None
        if start_x is None:
            if path:
                values=c["path_calibration_reset_positions"] if purpose=="calibration" else c["path_train_reset_positions"]
                start_x=float(np.random.default_rng(seed+97).choice(values))
            elif is_contact and c.get("contact_backend")=="ct_terrain":start_x=0.
            else:start_x=float(np.random.default_rng(seed+97).choice(c["contact_spawn_positions"])) if is_contact else 0.
        self.start_x=start_x
        from .contact import CONTACT_COHORTS
        if cohort not in (CONTACT_COHORTS if is_contact else ("iid","validation","regime","task","wind","joint")):
            raise ValueError(cohort)
        def choose(name):
            split="train"
            if c["case"]=="J":
                if cohort=="validation":split="validation"
                elif cohort=="joint" or (cohort=="regime" and name in ("friction","load")) or (cohort=="task" and name=="speed"):
                    split="test"
            values=c[name+"_"+split]
            return float(values[episode%len(values)] if name=="speed" else rng.choice(values))
        self.g=choose("speed")
        self._friction,self._load=choose("friction"),choose("load")
        if is_contact and not c["contact_robot_variation"]:self._friction,self._load=1.,0.
        self._wind=0.
        self._wind_kappa=c["wind_test_kappa"] if cohort in ("wind","joint") else c["wind_kappa"]
        self._wind_sigma=c["wind_test_sigma"] if cohort in ("wind","joint") and c["experiment"]=="wind" else c["wind_sigma"]
        selected_clock=clock or c["clock"]
        gamma_native=c["gamma_reference"]**(.01/c["reference_time"])
        self.base=make_common_env("cheetah-run",seed,"regular" if selected_clock=="fixed" else "irregular",gamma_native=gamma_native)
        b=self.base
        b.dt=c["fixed_hold"] if selected_clock=="fixed" else .01
        b.time_sampling="uniform" if selected_clock=="fixed" else "irregular"
        b.min_dt,b.max_dt=c["min_hold"],c["max_hold"]
        b.max_steps,b.episode_duration=c["episode_step_cap"],c["horizon"]
        p=b._env.physics
        self.contact_map=None;self.contact_audit=None
        compiled_xml=None
        if is_contact:
            from dm_control.suite import cheetah
            from .contact import sample_map,build_xml,validate_map,ContactAudit,query
            self.contact_map=environment_spec if environment_spec is not None else sample_map(c,seed,cohort,progress,difficulty)
            validate_map(self.contact_map)
            if self.contact_map["case"]!=c["case"] or self.contact_map["profile"]!=c["contact_profile"]:
                raise ValueError("Contact manifest and experiment differ")
            if self.contact_map.get("backend","mesh")!=c.get("contact_backend","mesh"):
                raise ValueError("Contact manifest and configured physical backend differ")
            xml,assets=cheetah.get_model_and_assets()
            compiled_xml=build_xml(xml,self.contact_map)
            p.reload_from_xml_string(compiled_xml,assets)
            if self.contact_map.get("backend")=="ct_terrain":
                from .terrain import apply_heightfields
                apply_heightfields(p,self.contact_map)
        if is_contact or path:
            original_initialize=b._env._task.initialize_episode
            height=float(query(self.contact_map,[start_x])["height"][0]) if is_contact else 0.
            def initialize(physics):
                physics.data.qpos[0]=start_x
                physics.data.qpos[1]=height
                original_initialize(physics)
            b._env._task.initialize_episode=initialize
        p.model.opt.timestep=c["physics_dt"]
        if (p.model.nq,p.model.nv,p.model.nu)!=(9,9,6):
            raise RuntimeError("Cheetah model dimensions differ from the notes")
        gears=p.model.actuator_gear[:,0].copy()
        if not np.allclose(gears,[120,90,60,90,60,30]):
            raise RuntimeError("Motor gears differ from the declared Cheetah actuator calibration")
        try:
            xml=importlib.resources.files("dm_control.suite").joinpath("cheetah.xml").read_bytes()
        except (FileNotFoundError,AttributeError):
            raise RuntimeError("Cannot record the installed Cheetah XML identity")
        self.backend=dict(xml_sha256=hashlib.sha256(xml).hexdigest(),gears=gears.tolist(),
                          nq=9,nv=9,nu=6,native_control_dt=b.dt_default,
                          mujoco_integrator=int(p.model.opt.integrator),
                          force_body=b.force_body_name,force_point_local=b.force_point_local.tolist())
        if is_contact:
            self.backend.update(contact_xml_sha256=hashlib.sha256(compiled_xml.encode()).hexdigest(),
                contact_map_sha256=self.contact_map["map_sha256"],contact_sensor=c["contact_sensor"],
                observation_dim=observation_dim(c),ground_priority=self.contact_map["priority"],contact_condim=3,
                contact_backend=self.contact_map.get("backend","mesh"))
            if self.contact_map.get("backend")=="ct_terrain":
                self.backend.update(terrain_upstream_commit=self.contact_map["upstream_commit"],
                    terrain_course_bounds=[self.contact_map["patch_edges"][0],self.contact_map["patch_edges"][-1]],
                    terrain_patches=len(self.contact_map["mu"]),terrain_grid_spacing=c["terrain_resolution"],
                    hfield_data_sha256=hashlib.sha256(p.model.hfield_data.tobytes()).hexdigest())
        p.model.geom_friction[:,0]*=self._friction
        torso=p.model.name2id("torso","body")
        mass=float(p.model.body_mass[torso])
        p.model.body_mass[torso]+=self._load
        p.model.body_inertia[torso]*=(mass+self._load)/mass
        mujoco.mj_setConst(p.model.ptr,p.data.ptr)
        # The known descriptor changes the reward law. It exposes no physical regime.
        def task_reward(physics):
            speed=physics.speed() if hasattr(physics,"speed") else physics.named.data.subtree_linvel["torso",0]
            if self.path_task is not None:
                # Convert a physical cost per second to the benchmark's
                # endpoint reward per reference interval.
                return -c["reference_time"]*self.path_task.cost_rate(speed,physics.data.ctrl)
            return float(np.clip(speed/self.g,0,1))
        b._env._task.get_reward=task_reward
        if is_contact:
            self.contact_audit=ContactAudit(p,self._friction)
        def after_physics(dt):
            if self.contact_audit is not None:self.contact_audit.sample(dt)
            if self.path_task is not None:
                x,speed,clearance,pitch=self._path_state()
                self.path_task.record(x,speed,p.data.ctrl,dt,clearance,pitch)
        if is_contact or path:b._after_physics_step=after_physics
        if path:
            def termination(physics):
                if self.path_task is None:return None
                x,_,clearance,pitch=self._path_state()
                return 0. if self.path_task.condition(x,clearance,pitch) else None
            b._env._task.get_termination=termination

        def wind_force(_):
            physics=b._env.physics
            physics.data.qfrc_applied[:]=0.
            physics.data.xfrc_applied[:]=0.
            if c["experiment"]!="wind":return 0.,0.,0
            body=b.force_body
            point=physics.data.xpos[body]+physics.data.xmat[body].reshape(3,3)@b.force_point_local
            mujoco.mj_jac(physics.model.ptr,physics.data.ptr,b._jacp,b._jacr,point,body)
            force=np.array([c["wind_force_scale"]*self._wind,0.,0.])
            tau=b._jacp.T@force
            physics.data.qfrc_applied[:]=tau
            power=float(force@(b._jacp@physics.data.qvel))
            # Wind has an exact OU marginal step. Mechanics sees the held force
            # for this substep. There is no white-noise momentum increment.
            self._wind=ou_step(self._wind,self._wind_kappa,self._wind_sigma,c["physics_dt"],self.wind_rng)
            return power,float(np.linalg.norm(tau)),int(np.count_nonzero(abs(tau)>1e-10))
        b._point_force=types.MethodType(wind_force,b)
        b.reset(seed=seed)
        if path:
            from .path_task import PathTask
            # Course reset positions refer to the same subtree COM used by
            # the speed sensor. Preserve root clearance when shifting a pose
            # from the source's settling procedure onto the sampled terrain.
            root_x=float(p.data.xpos[torso,0]);delta=start_x-float(p.data.subtree_com[torso,0])
            p.data.qpos[0]+=delta
            if self.contact_map is not None:
                from .contact import query
                heights=query(self.contact_map,[root_x,root_x+delta])["height"]
                p.data.qpos[1]+=float(heights[1]-heights[0])
            p.data.qvel[:]=0.
            mujoco.mj_forward(p.model.ptr,p.data.ptr)
            if purpose=="calibration" and initial_speed is None:
                initial_speed=self.g*float(np.random.default_rng(seed+101).choice(c["path_calibration_velocity_scales"]))
            if initial_speed is not None:
                p.data.qvel[0]+=initial_speed-float(p.speed())
                mujoco.mj_forward(p.model.ptr,p.data.ptr)
            self.path_task=PathTask(c,self.g,float(p.data.subtree_com[torso,0]))
            _,v,clearance,_=self._path_state()
            self.path_task.seed_trace(v,clearance)
        self.observation=self._observe()

    def _path_state(self):
        p=self.base._env.physics;torso=p.model.name2id("torso","body")
        root=p.data.xpos[torso]
        height=0.
        if self.contact_map is not None:
            from .contact import query
            height=float(query(self.contact_map,[root[0]])["height"][0])
        return float(p.data.subtree_com[torso,0]),float(p.speed()),float(root[2]-height),float(p.data.qpos[2])

    def _observe(self):
        p=self.base._env.physics
        positions=p.data.qpos[1:].copy()
        if self.config["sensor"]=="position":
            observed=(positions+self.sensor_rng.normal(0,self.config["position_std"],8)).astype(np.float32)
        else:observed=np.concatenate([positions,p.data.qvel.copy()]).astype(np.float32)
        if self.contact_map is not None:
            from .contact import measure
            torso=p.model.name2id("torso","body")
            root=p.data.xpos[torso]
            observed=np.concatenate([observed,measure(self.config,self.contact_map,root[0],root[2],
                self.environment_sensor_rng,self._friction)])
        if self.path_task is not None:
            x=self._path_state()[0]+self.odometry_rng.normal(0,self.config["path_odometry_std"])
            observed=np.concatenate([observed,[(self.path_task.goal-x)/self.config["path_length"]]])
        if observed.shape!=(observation_dim(self.config),) or not np.isfinite(observed).all():
            raise FloatingPointError("Invalid declared measurement vector")
        return observed

    def hold(self):
        requested=self.base._next_dt_request()
        if requested<=0:return self.config["fixed_hold"]
        return self.config["physics_dt"]*max(1,round(requested/self.config["physics_dt"]))

    def step(self,action):
        if self.path_task is not None and self.path_task.event_paid:raise RuntimeError("Reset after the finite path episode ends")
        if self.contact_audit is not None:self.contact_audit.reset()
        _,reward,term,trunc,info=self.base.step(action)
        self.observation=self._observe()
        public=dict(r_src=info["endpoint_reward_rate"],R=reward*self.base.dt_default/self.config["reference_time"],
                    h=info["dt_used"],next_h=self.hold(),terminated=term,truncated=trunc,
                    physics_steps=info["physics_steps"],reward_time_integral=info["reward_time_integral"],
                    external_work=info["external_work"],directly_forced_dofs=info["directly_forced_dofs"])
        if self.path_task is not None:
            event=self.path_task.close_interval(trunc)
            gamma=self.config["gamma_reference"]**(public["h"]/self.config["reference_time"])
            public["R"]+=gamma*event
            public["r_src"]+=gamma*event/(public["h"]/self.config["reference_time"])
            public["path_event_reward"]=event
            public["path_metrics"]=self.path_task.summary(include_trace=False)
        if self.contact_audit is not None:
            # These fields are evaluation diagnostics. Runner and calibration
            # select their permitted replay fields explicitly.
            public["contact_audit"]=self.contact_audit.summary()
            from .contact import query
            p=self.base._env.physics
            root=p.data.xpos[p.model.name2id("torso","body")]
            clearance=float(root[2]-query(self.contact_map,[root[0]])["height"][0])
            public["contact_audit"].update(root_x=float(root[0]),torso_clearance=clearance,
                low_torso=bool(clearance<self.config["contact_fall_height"]))
            if self.contact_map.get("backend")=="ct_terrain":
                edges=self.contact_map["patch_edges"]
                public["contact_audit"]["off_course"]=not edges[0]<=root[0]<=edges[-1]
        return self.observation,public

    def audit_truth(self):
        # Evaluation audit only. No training record or network input uses this map.
        result=dict(friction=self._friction,load_kg=self._load,wind=self._wind,
                    wind_kappa=self._wind_kappa,wind_sigma=self._wind_sigma)
        if self.contact_map is not None:
            result.update(contact_map_sha256=self.contact_map["map_sha256"],map_bank=self.contact_map["bank"],
                map_index=self.contact_map["index"],map_difficulty=self.contact_map["difficulty"],start_x=self.start_x)
        return result

    def close(self):self.base.close()
