"""Case definitions, causal interfaces, and experiment identities."""
from copy import deepcopy
import json
import math
import re
from pathlib import Path
from bench.protocol import ROOT, SOURCE_SHA256, digest

METHODS = ("ct_sac", "ct_sac_i", "ct_td3", "interval_sac", "model_sac",
           "interval_euler", "ph_sac", "ph_direct")
DEFAULT_EXPERIMENTS = {"F": ("native", "position"), "J": ("static", "wind")}
EXPERIMENTS = {k: v + ("contact",) for k, v in DEFAULT_EXPERIMENTS.items()}
CONFIG_VERSION = 4
TERRAIN_REFERENCE_COMMIT="297ecbedb1b7510439f85927aa4b70d893f681e8"
ALGORITHM_SETS={"core":("ct_sac","interval_euler","ph_sac"),
                "ct":("ct_sac","ct_sac_i","ct_td3"),
                "solver":("model_sac","interval_euler","ph_sac","ph_direct"),"all":METHODS}


def select_methods(methods=None,algorithm_set="core"):
    if algorithm_set not in ALGORITHM_SETS:raise ValueError("Unknown algorithm set")
    values=re.split(r"[,\s]+",methods.strip()) if methods is not None else list(ALGORITHM_SETS[algorithm_set])
    if not values or any(x not in METHODS for x in values):raise ValueError(f"Choose algorithms from {METHODS}")
    return list(dict.fromkeys(values))


def is_path_task(c):return c.get("control_task","native_run")=="path_speed"


def make_config(case="F", experiment=None, seed=0, overrides=None):
    if case not in EXPERIMENTS:
        raise ValueError("Choose F or J")
    experiment = experiment or EXPERIMENTS[case][0]
    if experiment not in EXPERIMENTS[case]:
        raise ValueError(f"{case} experiments: {EXPERIMENTS[case]}")
    path=(overrides or {}).get("control_task","native_run")=="path_speed"
    target=float((overrides or {}).get("path_target_speed",3.))
    length=float((overrides or {}).get("path_length",20.));start=float((overrides or {}).get("path_start_x",0.))
    c = dict(schema=CONFIG_VERSION, case=case, experiment=experiment, seed=int(seed),
        task="cheetah-run", branch="Op", source_sha256=SOURCE_SHA256,
        sensor="position" if experiment == "position" else "native", position_std=.01,
        history=16, encoder_dim=32, use_reward_history=True,
        speed_train=[target] if path else ([10.] if case == "F" else [6., 10., 14.]),
        friction_train=[1.] if case == "F" else [.6, 1., 1.4],
        load_train=[0.] if case == "F" else [0., 1., 2.],
        speed_validation=[target] if path else ([10.] if case == "F" else [8., 12.]),
        friction_validation=[1.] if case == "F" else [.8, 1.2],
        load_validation=[0.] if case == "F" else [.5, 1.5],
        speed_test=([target] if case=="F" else [.9*target,1.1*target]) if path else ([10.] if case == "F" else [7., 13.]),
        friction_test=[1.] if case == "F" else [.7, 1.3],
        load_test=[0.] if case == "F" else [.25, 1.75],
        wind_kappa=1., wind_sigma=.4 if experiment == "wind" else 0.,
        wind_test_kappa=.5, wind_test_sigma=.7, wind_force_scale=3.,
        clock="irregular", fixed_hold=.01, physics_dt=.002, min_hold=.002, max_hold=.03,
        horizon=12. if path else 10., episode_step_cap=6000 if path else 2000, reference_time=.01,
        gamma_reference=math.exp(-.05*.01) if path else .98,
        entropy_convention="start", alpha_reference=.002 if path else .2, boundary="finite_horizon" if path else "continuing",
        total_decisions=1_000_000, calibration_decisions=30_000,
        calibration_protocol="goal_balanced_v2" if path else "episode_v1",
        path_calibration_goal_episodes=4,path_calibration_goal_attempts=32,path_calibration_probe_steps=4,
        observer_updates=2000, operator_updates=5000, batch=256, capacity=300_000,
        hidden=[400,300], lr=3e-4, tau=.005, action_samples=3, nodes=96,
        ensembles=1 if case == "F" else 3, kernel_temperature=.5,
        probe_count=8, probe_weight=1., closure_weight=.1,
        phase_steps=3, sigma=.5, damping=1., omega=3., euler_eta=.5,
        inverse_tolerance=1e-8,inverse_relative_tolerance=1e-12,inverse_refinement_steps=2,residual_tolerance=1e-4,momentum_tolerance=1e-4,
        require_solver_tolerance=False,fit_admission="empirical",fit_steps=4,fit_relative_tolerance=.1,
        fit_storage_allowance=.25, log_every=1000, save_every=100_000,
        model_update="frozen",online_model_every=100,operator_validation_growth=.05,td3_delay=2,
        td3_exploration=.1, td3_target_noise=.2, td3_noise_clip=.5,
        contact_backend="mesh",contact_profile="mixed", contact_sensor="geometry", contact_robot_variation=False,
        terrain_reference_commit=TERRAIN_REFERENCE_COMMIT,
        terrain_elevation="hills",terrain_course_start=-5.,terrain_course_length=200.,
        terrain_patch_length=4.,terrain_resolution=.05,terrain_max_elevation=1.,
        terrain_start_flat=3.,terrain_ramp_in=5.,terrain_hill_height=.3,terrain_hill_wavelength=4.,
        terrain_max_grade=.25,terrain_slope_segment=[2.,8.],terrain_step_height=.1,terrain_step_segment=[1.,4.],
        terrain_friction_min=.1,terrain_friction_max=1.,
        contact_fixed_seed=1729, contact_train_maps=32, contact_validation_maps=8, contact_test_maps=16,
        contact_patch_length=1., contact_patches=240, contact_half_width=1.5,
        contact_spawn_positions=[0.,4.,8.,12.,16.],
        contact_amplitude_train=[.08,.24], contact_amplitude_test=[.30,.40],
        contact_wavelength_train=[5.,9.], contact_wavelength_test=[3.5,4.5],
        contact_mu_train=[.35,1.0], contact_mu_test=[.12,.25],
        contact_tau_train=[.02,.05], contact_tau_test=[.06,.08],
        contact_zeta_train=[.8,1.3], contact_zeta_test=[.45,.65],
        contact_offsets=[-1.,-.5,0.,.5,1.,2.,3.],
        contact_height_std=.005, contact_slope_std=.01,
        contact_fall_height=.25,
        contact_mu_std=.03, contact_tau_std=.002, contact_zeta_std=.03,
        contact_curriculum=(case=="J" and experiment=="contact" and (overrides or {}).get("contact_backend","mesh")=="mesh"),
        contact_stage_boundaries=[0.,.15,.4], contact_stage_levels=[.35,.65,1.],
        contact_calibration_levels=[1.] if (overrides or {}).get("contact_backend")=="ct_terrain" else [.35,.65,1.],
        control_task="path_speed" if path else "native_run",path_target_speed=target,
        path_start_x=start,path_length=length,path_train_reset_positions=[start],
        path_calibration_reset_positions=[start+f*length for f in (0.,.2,.4,.6,.8,.975)],path_calibration_velocity_scales=[0.,.5,1.],
        path_speed_weight=4.,path_effort_weight=.01,path_time_weight=.05,path_backwards_weight=.1,
        path_goal_bonus=5.,path_failure_penalty=2.,path_speed_band=.1,path_odometry_std=.01,
        path_fall_clearance=.20,path_fall_pitch=1.5,path_trace_dt=.05)
    if overrides:
        unknown = set(overrides) - set(c)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        c.update(deepcopy(overrides))
    if c["case"] != case or c["experiment"] != experiment or c["schema"] != CONFIG_VERSION or c["seed"]!=int(seed):
        raise ValueError("Case identity cannot be overridden")
    if c["task"] != "cheetah-run" or c["branch"] != "Op":
        raise ValueError("The new F/J implementation is the Cheetah Op branch in Section 7.7")
    if not 0 <= c["seed"] < 3000 or c["sensor"] not in ("native", "position"):
        raise ValueError("Invalid seed or measurement interface")
    if c["clock"] not in ("irregular", "fixed") or c["entropy_convention"] not in ("start", "discounted"):
        raise ValueError("Declare a supported clock and entropy convention")
    if c["boundary"] not in ("continuing", "finite_horizon") or c["model_update"] not in ("frozen", "online"):
        raise ValueError("Invalid boundary or model-update contract")
    if c["fit_admission"] not in ("empirical","strict"):
        raise ValueError("Choose empirical distillation or strict returned-storage admission")
    if c["boundary"]=="finite_horizon" and c["episode_step_cap"]<int(math.ceil(c["horizon"]/c["min_hold"])):
        raise ValueError("A time-only finite horizon needs episode_step_cap >= ceil(horizon/min_hold)")
    if not 0 < c["gamma_reference"] < 1 or c["alpha_reference"] <= 0:
        raise ValueError("Discount and entropy constants must be positive")
    if not c["physics_dt"] <= c["min_hold"] <= c["max_hold"]:
        raise ValueError("Invalid physical and decision times")
    if not c["min_hold"] <= c["fixed_hold"] <= c["max_hold"]:
        raise ValueError("Fixed hold is outside the declared range")
    if c["horizon"]<=0 or c["episode_step_cap"]<1:raise ValueError("Positive deadline and decision cap required")
    if min(c[k] for k in ("history","batch","nodes","action_samples","ensembles","phase_steps",
                          "fit_steps","save_every","log_every","probe_count","online_model_every")) <= 0:
        raise ValueError("Counts must be positive")
    if c["nodes"] < len(c["speed_train"]) or c["calibration_decisions"] < c["batch"]:
        raise ValueError("Insufficient calibration data or nodes")
    if c["total_decisions"] <= c["calibration_decisions"]:
        raise ValueError("The interaction budget includes calibration and must leave policy data")
    if c["damping"] <= 0 or c["sigma"] <= 0 or not 0 < c["euler_eta"] <= 1:
        raise ValueError("The numerical step requires damping and positive time")
    if (not math.isfinite(c['inverse_tolerance']) or not math.isfinite(c['inverse_relative_tolerance']) or
        c['inverse_tolerance']<0 or c['inverse_relative_tolerance']<0 or
        not isinstance(c['inverse_refinement_steps'],int) or not 0<=c['inverse_refinement_steps']<=5):
        raise ValueError('Invalid inverse recovery tolerance or refinement count')
    if case == "F" and ((not path and c["speed_train"] != [10.]) or c["friction_train"] != [1.] or c["load_train"] != [0.] or c["wind_sigma"] != 0):
        raise ValueError("F fixes the task, ground, load, and environmental law")
    if c["control_task"] not in ("native_run","path_speed"):raise ValueError("Unknown control task")
    if path:
        if c["calibration_protocol"]!="goal_balanced_v2":raise ValueError("Use the current path calibration protocol")
        counts=("path_calibration_goal_episodes","path_calibration_goal_attempts","path_calibration_probe_steps")
        if any(not isinstance(c[k],int) or isinstance(c[k],bool) or c[k]<1 for k in counts):raise ValueError("Positive integer calibration probe counts required")
        if c["path_calibration_goal_episodes"]<2 or c["path_calibration_goal_attempts"]<c["path_calibration_goal_episodes"]:
            raise ValueError("Collect at least two goal episodes per task, with enough bounded attempts")
        if c["calibration_decisions"]<c["path_calibration_goal_episodes"]*len(c["speed_train"])+1:
            raise ValueError("Calibration budget must include goal probes and ordinary transitions")
        if c["boundary"]!="finite_horizon":raise ValueError("The finite path task includes the deadline in its information state")
        if c["path_length"]<=0 or c["path_target_speed"]<=0 or c["path_trace_dt"]<=0 or c["path_odometry_std"]<0:
            raise ValueError("Invalid path, speed, trace clock, or odometry settings")
        if any(x<=0 for k in ("speed_train","speed_validation","speed_test") for x in c[k]):raise ValueError("Target speeds must be positive")
        if case=="F" and any(c[k]!=[c["path_target_speed"]] for k in ("speed_train","speed_validation","speed_test")):
            raise ValueError("F fixes one known target speed across all splits")
        for key in ("path_train_reset_positions","path_calibration_reset_positions","path_calibration_velocity_scales"):
            if not c[key] or any(not math.isfinite(x) for x in c[key]):raise ValueError("Invalid path reset distribution")
        goal=c["path_start_x"]+c["path_length"]
        if any(x>=goal for x in c["path_train_reset_positions"]):raise ValueError("Policy resets must start before the goal")
        if any(x>=goal for x in c["path_calibration_reset_positions"]):raise ValueError("Calibration resets must start before the goal")
        if any(c[k]<0 for k in ("path_speed_weight","path_effort_weight","path_time_weight","path_backwards_weight","path_goal_bonus","path_failure_penalty")):
            raise ValueError("Path cost weights must be nonnegative")
    if case == "J":
        for name in ("speed", "friction", "load"):
            if set(c[name+"_train"]) & set(c[name+"_test"]):
                raise ValueError(f"J test {name} values must be held out")
    if c["contact_profile"] not in ("elevation","friction","material","mixed"):
        raise ValueError("Unknown contact profile")
    if c["contact_backend"] not in ("mesh","ct_terrain"):raise ValueError("Unknown contact backend")
    if c["contact_backend"]=="ct_terrain":
        if c["terrain_reference_commit"]!=TERRAIN_REFERENCE_COMMIT:raise ValueError("Unsupported CT terrain source commit")
        if experiment!="contact":raise ValueError("The CT terrain backend requires experiment=contact")
        if c["contact_profile"]=="material":raise ValueError("CT terrain uses stock contact response; choose elevation, friction, or mixed")
        if c["contact_curriculum"] or c["contact_calibration_levels"]!=[1.] or c["contact_robot_variation"]:
            raise ValueError("Matched CT terrain uses full courses, calibration level [1], and the stock robot")
        if c["terrain_elevation"] not in ("hills","slopes","steps"):raise ValueError("Choose hills, slopes, or steps")
        positive=("terrain_patch_length","terrain_resolution","terrain_max_elevation","terrain_hill_wavelength","terrain_friction_min")
        if any(not math.isfinite(c[k]) or c[k]<=0 for k in positive):raise ValueError("Positive finite terrain scales required")
        if not c["terrain_course_start"]<0<c["terrain_start_flat"]<c["terrain_course_length"]:
            raise ValueError("Require course_start < 0 < start_flat < course_length")
        if any(not math.isfinite(c[k]) for k in ("terrain_course_start","terrain_start_flat","terrain_course_length")):
            raise ValueError("Finite terrain course bounds required")
        if not c["terrain_friction_min"]<=c["terrain_friction_max"] or not math.isfinite(c["terrain_friction_max"]):
            raise ValueError("Invalid terrain friction interval")
        cells=c["terrain_patch_length"]/c["terrain_resolution"]
        if abs(cells-round(cells))>1e-6 or cells<1:raise ValueError("Terrain patch length must be a multiple of resolution")
        for key in ("terrain_slope_segment","terrain_step_segment"):
            if len(c[key])!=2 or not 0<c[key][0]<=c[key][1] or any(not math.isfinite(x) for x in c[key]):
                raise ValueError("Invalid terrain segment lengths")
        if any(not math.isfinite(c[k]) or c[k]<0 for k in ("terrain_ramp_in","terrain_hill_height","terrain_max_grade","terrain_step_height")):
            raise ValueError("Invalid terrain amplitude, grade, step, or blend")
        if c["terrain_hill_height"]>c["terrain_max_elevation"]:raise ValueError("Hills must fit within the heightfield band")
        if not 0<=c["contact_fixed_seed"]<2**32:raise ValueError("Terrain seed must fit RandomState")
        if path and (c["path_start_x"]<c["terrain_course_start"] or goal>c["terrain_course_length"]):
            raise ValueError("The finite path must fit inside the terrain course")
    if c["contact_sensor"] not in ("history","geometry","material"):
        raise ValueError("Choose history, geometry, or the declared synthetic material sensor")
    if case=="F" and c["contact_curriculum"]:
        raise ValueError("F keeps one physical map; use J for a map curriculum")
    for key in ("amplitude","wavelength","mu","tau","zeta"):
        for split in ("train","test"):
            a,b=c[f"contact_{key}_{split}"]
            if not 0<a<=b or not all(math.isfinite(x) for x in (a,b)):
                raise ValueError(f"Invalid contact {key} range")
    if c["contact_tau_train"][0]<2*c["physics_dt"] or c["contact_tau_test"][0]<2*c["physics_dt"]:
        raise ValueError("Positive contact time constants must exceed the two-step refsafe bound")
    if c["contact_patch_length"]<=0 or c["contact_patches"]<16 or c["contact_half_width"]<=0:
        raise ValueError("Invalid terrain grid")
    if min(c[k] for k in ("contact_train_maps","contact_validation_maps","contact_test_maps"))<1:
        raise ValueError("Each map bank needs at least one map")
    if not c["contact_offsets"] or any(not math.isfinite(x) for x in c["contact_offsets"]):
        raise ValueError("Declare finite sensor offsets")
    if not c["contact_spawn_positions"] or any(not math.isfinite(x) for x in c["contact_spawn_positions"]):
        raise ValueError("Declare finite contact reset positions")
    bounds,levels=c["contact_stage_boundaries"],c["contact_stage_levels"]
    if len(bounds)!=len(levels) or not bounds or bounds[0]!=0 or bounds!=sorted(set(bounds)) or bounds[-1]>=1:
        raise ValueError("Curriculum boundaries must start at zero and increase below one")
    if any(not 0<x<=1 for x in levels+c["contact_calibration_levels"]) or not c["contact_calibration_levels"]:
        raise ValueError("Curriculum levels must lie in (0,1]")
    for name in ("height","slope","mu","tau","zeta"):
        if c[f"contact_{name}_std"]<0:raise ValueError("Sensor standard deviations must be nonnegative")
    return c


def observation_dim(c):
    size=8 if c["sensor"]=="position" else 17
    if c["experiment"]=="contact" and c["contact_sensor"]!="history":
        size+=len(c["contact_offsets"])*(5 if c["contact_sensor"]=="material" else 2)
    return size+int(is_path_task(c))


def config_from(path=None, case="F", experiment=None, seed=0,extra=None):
    override = json.loads(Path(path).read_text()) if path else {}
    if extra:override.update(extra)
    return make_config(case, experiment, seed, override)


def calibration_key(c):
    # Critic budgets and solver settings do not change a common calibration artifact.
    excluded = {"phase_steps","sigma","damping","omega","euler_eta","fit_steps","inverse_relative_tolerance","inverse_refinement_steps",
                "fit_relative_tolerance","fit_storage_allowance","inverse_tolerance",
                "residual_tolerance","momentum_tolerance","require_solver_tolerance","fit_admission",
                "total_decisions","model_update","online_model_every","operator_validation_growth","tau","save_every","log_every"}
    return f"{c['case']}_{c['experiment']}_seed{c['seed']}_" + digest({k:v for k,v in c.items() if k not in excluded})[:12]


def run_key(c, method):
    if method not in METHODS:
        raise ValueError(method)
    return f"FJ/{c['case']}/{c['experiment']}/{method}/seed{c['seed']}_" + digest(dict(config=c,method=method))[:12]


def label(c, method):
    if method == "ph_sac":
        return "CPHT-SAC" if c["ensembles"] == 1 else "TPHT-SAC"
    return method


def source_record():
    return json.loads((ROOT / "cases/sources.json").read_text())
