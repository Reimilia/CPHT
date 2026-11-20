"""Calibration, bounded training, checkpoint recovery, and paired F/J evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch

from bench.protocol import digest, write_json, versions
from bench.training import atomic_torch_save
from physical_fj.mechanics import NominalRobot, PhysicalEnvironment
from physical_fj.prepare import load as load_calibration, prepare as prepare_calibration
from pure_cpht.learning import prepare_artifact
from environment_belief.config import contract
from environment_belief.runtime import summarize as summarize_assimilation
from .config import paths
from .controller import ComparisonController
from .learning import Trainer, calibration_replay

def preparation(profile, job, run_root, artifact_root):
    s, c, artifact, run = paths(profile, job, run_root, artifact_root)
    report = prepare_calibration(s, c, artifact)
    prepared = prepare_artifact(artifact, s)
    result = dict(schema=profile["schema"], artifact=str(artifact),
                  calibration_id=report["calibration_id"],
                  model_sha256=digest(prepared["model"]), frozen_model=True,
                  mechanics=prepared["metrics"]["mechanics"],
                  contact=prepared["metrics"]["contact"],
                  physical_calibration_decisions=report["counters"]["decisions"])
    write_json(artifact / "comparison_preparation.json", result)
    return result

def make_controller(saved, seed, device, gait=None):
    s = saved["settings"]
    return ComparisonController(
        saved["pure_model"], s["pure"], saved["neutral"], saved["control"],
        saved["profile"]["optimization"], saved["job"]["horizon"], saved["job"]["method"],
        saved["normalization"], seed, device, gait)

def initial_saved(profile, job, s, c, artifact):
    calibration, data, norm = load_calibration(artifact)
    prepared = prepare_artifact(artifact, s)
    return dict(schema=profile["schema"], profile=deepcopy(profile), job=dict(job),
                settings=s, control=c, pure_model=prepared["model"], neutral=prepared["neutral"],
                model_metrics=prepared["metrics"], normalization=norm, calibration=calibration,
                model_sha256=digest(prepared["model"]), versions=versions(),
                information_contract=contract(s), optimization_scope="motion coefficients / selector / task value",
                model_parameters_frozen_after_calibration=True), data

def training(profile, job, run_root, artifact_root, device="cpu", resume=False, pause_episodes=None):
    s, c, artifact, run = paths(profile, job, run_root, artifact_root)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "final.pt").exists() and not resume:
        return dict(status="existing_complete_checkpoint", checkpoint=str(run / "final.pt"))
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    saved, data = initial_saved(profile, job, s, c, artifact)
    prototype = make_controller(saved, job["seed"], device)
    trainer = Trainer(prototype, profile["optimization"]) if prototype.selector is not None else None
    bank = calibration_replay(data, saved["normalization"], len(prototype.robot.point_body),
                              profile["optimization"]["max_replay"], job["seed"] + 173)
    rng = np.random.default_rng(job["seed"] + 823)
    gait = None
    counters = dict(decisions=s["calibration_decisions"], policy_decisions=0, physics_steps=0,
                    physical_seconds=0.0, selector_updates=0, critic_updates=0,
                    skipped_updates=0, control_plans=0, predictive_substeps=0,
                    mechanical_weight_updates=0, checkpoint_wall_seconds=0.0)
    episode = 0
    if resume:
        recovered = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        for key in ("job", "settings", "control", "model_sha256"):
            if recovered[key] != saved[key]:
                raise ValueError(f"Resume changed {key}; select a new run root")
        if recovered["profile"]["optimization"] != profile["optimization"]:
            raise ValueError("Resume changed optimizer/rollout settings")
        prototype.load_learned(recovered["learned"])
        if trainer:
            trainer.restore(recovered["trainer"])
        bank.restore(recovered["replay"])
        counters, episode, gait = recovered["counters"], recovered["next_episode"], recovered["gait"]
        rng.bit_generator.state = recovered["numpy_rng"]
        torch.set_rng_state(recovered["torch_rng"])
    elif (run / "latest.pt").exists():
        raise FileExistsError("Partial run exists; use --resume")
    write_json(run / "run.json", dict(profile=profile, job=job, settings=s, control=c,
               information_contract=contract(s), model_sha256=saved["model_sha256"],
               calibration_id=saved["calibration"]["calibration_id"],
               learning_scope=saved["optimization_scope"], smoke_only=profile["smoke_only"]))
    stop = {"requested": False}
    def stopping(*_):
        stop["requested"] = True
    handlers = {sig: signal.getsignal(sig) for sig in
                set((signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)))}
    for sig in handlers:
        signal.signal(sig, stopping)
    started = time.perf_counter()
    first_episode = episode
    last_save = counters["decisions"]
    last_stats = {}
    current = prototype
    def save(path, recovery=False):
        packet = dict(saved, learned=prototype.learned_state(), gait=gait,
                      counters=dict(counters), trained_selector=bool(counters["selector_updates"]),
                      smoke_only=profile["smoke_only"])
        if recovery:
            packet.update(trainer=trainer.state() if trainer else None, replay=bank.state(),
                          next_episode=episode, numpy_rng=rng.bit_generator.state,
                          torch_rng=torch.get_rng_state())
        atomic_torch_save(packet, path)
    try:
        with (run / "train.jsonl").open("a", buffering=1) as log:
            while counters["decisions"] < s["total_decisions"] and not stop["requested"]:
                env = PhysicalEnvironment(c, s, 1000000000 + job["seed"] * 10000 + episode,
                                          episode, nominal=NominalRobot(s), purpose="train")
                current = make_controller(saved, job["seed"] * 100000 + episode, device, gait)
                if trainer:
                    current.selector, current.value = trainer.selector, trainer.value
                try:
                    while counters["decisions"] < s["total_decisions"]:
                        before = env.current
                        action = current.act(before, env.g, env.hold(), env.base.cur_t)
                        bank.add(current.current_snapshot())
                        sigma = profile["optimization"]["action_noise"]
                        executed = np.clip(action + rng.normal(0, sigma, 6), -1, 1).astype(np.float32)
                        _, info, _ = env.step(executed)
                        counters["decisions"] += 1
                        counters["policy_decisions"] += 1
                        counters["physics_steps"] += info["physics_steps"]
                        counters["physical_seconds"] += info["h"]
                        if trainer and counters["policy_decisions"] % profile["optimization"]["update_every"] == 0:
                            for _ in range(profile["optimization"]["updates_per_block"]):
                                last_stats = trainer.update(current, bank)
                            counters.update(selector_updates=trainer.updates, critic_updates=trainer.value_updates,
                                            skipped_updates=trainer.skipped)
                        if counters["decisions"] % s["log_every"] == 0:
                            log.write(json.dumps(dict(counters=counters, update=last_stats,
                                      controller=current.diagnostics)) + "\n")
                        if info["terminated"] or info["truncated"]:
                            break
                    gait = current.mean.detach().cpu().tolist()
                    with (run / "episodes.jsonl").open("a") as out:
                        out.write(json.dumps(dict(episode=episode, decisions=counters["decisions"],
                                  path=env.path_task.summary(False), controller=current.diagnostics)) + "\n")
                finally:
                    counters["control_plans"] += current.plans
                    counters["predictive_substeps"] += current.predicted_steps
                    env.close()
                episode += 1
                counters["checkpoint_wall_seconds"] = time.perf_counter() - started
                if pause_episodes and episode - first_episode >= pause_episodes:
                    stop["requested"] = True
                if counters["decisions"] - last_save >= s["save_every"] or stop["requested"]:
                    save(run / "latest.pt", True)
                    last_save = counters["decisions"]
        counters["checkpoint_wall_seconds"] = time.perf_counter() - started
        save(run / "latest.pt", True)
        complete = counters["decisions"] >= s["total_decisions"]
        if complete:
            save(run / "final.pt")
        result = dict(complete=complete, counters=counters, checkpoint=str(run / "final.pt") if complete else None)
        write_json(run / "status.json", result)
        return result
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

def scenario(saved, test_case, condition):
    from fjbench.config import make_config
    s, c = deepcopy(saved["settings"]), deepcopy(saved["control"])
    c["case"] = test_case
    if test_case != saved["control"]["case"]:
        base = make_config(test_case, "contact", c["seed"],
                           dict(control_task="path_speed", contact_backend="ct_terrain",
                                path_target_speed=c["path_target_speed"]))
        for field in ("speed", "friction", "load"):
            for split in ("train", "validation", "test"):
                c[field + "_" + split] = base[field + "_" + split]
        c["ensembles"] = base["ensembles"]
    c["contact_curriculum"] = False
    cohort = "layout" if condition == "layout" else "iid"
    if condition == "steps":
        c.update(contact_profile="mixed", terrain_elevation="steps")
    if condition == "friction_change":
        if saved["profile"]["smoke_only"]:
            events = [dict(time=0.02, scale=0.4), dict(time=0.06, scale=1.0)]
            s["sensing"].update(friction_events=events, friction_event_jitter=0., friction_scale_jitter=0.)
        else:
            events = [dict(time=3.0, scale=0.3), dict(time=7.0, scale=0.8)]
            s["sensing"].update(friction_events=events, friction_event_jitter=0.6, friction_scale_jitter=0.15)
    if condition == "sensor_shift":
        if s["sensing"]["mode"] != "sensed_belief":
            raise ValueError("Sensor-shift condition requires external geometric sensing")
        s["sensing"].update(sensor_delay=0.06, sensor_dropout=0.20)
    return s, make_config(test_case, "contact", c["seed"], c), cohort

def evaluate(checkpoint, test_case, clock, condition, device="cpu", overwrite=False, evaluation=None):
    path = Path(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "cpht_comparison_v1":
        raise ValueError("Use a new comparison checkpoint; legacy runners remain separate")
    profile, job = saved["profile"], saved["job"]
    if evaluation is not None:
        profile = deepcopy(profile)
        profile["evaluation"] = deepcopy(evaluation)
        saved = dict(saved, profile=profile)
    s, c, cohort = scenario(saved, test_case, condition)
    from .report import evaluation_signature
    tag = f"{job['case']}to{test_case}_{clock}_{condition}_{evaluation_signature(profile)}"
    output = path.parent / "evaluations" / f"{tag}.json"
    if output.exists() and not overwrite:
        prior = json.loads(output.read_text())
        if prior.get("diagnostic_version") != "relative_contact_residual_v1":
            raise FileExistsError("Stale diagnostics; pass --overwrite to reevaluate")
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if prior["checkpoint_sha256"] != checkpoint_hash:
            raise FileExistsError("Evaluation belongs to an older checkpoint; pass --overwrite")
        return prior
    torch.set_num_threads(1)
    rows = []
    for i in range(profile["evaluation"]["episodes"]):
        episode_seed = 100000 + i
        env = PhysicalEnvironment(c, s, episode_seed, i, cohort, clock, purpose="evaluation")
        ctl = make_controller(saved, episode_seed, device, saved["gait"])
        ctl.load_learned(saved["learned"])
        if ctl.selector is not None:
            ctl.selector.eval().requires_grad_(False)
        if ctl.value is not None:
            ctl.value.eval().requires_grad_(False)
        predictions, assimilation, decisions = [], [], []
        step = 0
        discounted_return = 0.
        discount = 1.
        error = None
        budget_censored = False
        recorder = None
        before_model_hash = digest(saved["pure_model"])
        try:
            if i < profile["evaluation"]["videos"]:
                from fjbench.recording import EpisodeRecorder
                recorder = EpisodeRecorder(env, path.parent / "videos" / f"{tag}_{i}.mp4",
                    dict(width=640, height=480, fps=25, overwrite=overwrite, end_card_seconds=1.),
                    f"CPHT {job['method']} | {job['case']} to {test_case}")
            while True:
                before = env.current
                action = ctl.act(before, env.g, env.hold(), env.base.cur_t)
                if not np.isfinite(action).all() or abs(action).max() > 1.00001:
                    raise FloatingPointError("Nonfinite or infeasible motor command")
                after, info, _ = env.step(action)
                discounted_return += discount * info["R"]
                discount *= c["gamma_reference"] ** (info["h"] / c["reference_time"])
                predictions.append(ctl.prediction(before, after, action, info["h"]))
                decisions.append(dict(ctl.diagnostics, time=env.base.cur_t, actual_hold=info["h"]))
                if env.sensing is not None:
                    assimilation.append(env.sensing.audit())
                step += 1
                if info["terminated"] or info["truncated"]:
                    break
                cap = profile["evaluation"]["max_decisions"]
                if cap is not None and step >= cap:
                    budget_censored = True
                    break
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            path_metrics = env.path_task.summary()
            row = dict(episode_seed=episode_seed, target_speed=env.g,
                       audit_truth=env.audit_truth(), path_metrics=path_metrics,
                       discounted_task_return=discounted_return, decisions=step,
                       budget_censored=budget_censored, error=error,
                       predictions=predictions, controller_trace=decisions,
                       failure_counts=dict(ctl.failure_counts), wall_clock_plans=ctl.plans,
                       predictive_substeps=ctl.predicted_steps, physical_seconds=env.base.cur_t,
                       model_unchanged=(before_model_hash == digest(saved["pure_model"])))
            if assimilation:
                row.update(assimilation=summarize_assimilation(assimilation),
                           assimilation_trace=assimilation)
            if recorder:
                row["video"] = recorder.close(error is None, row)
            env.close()
        from .report import failure_flags
        row["failure_flags"] = failure_flags(row, profile)
        rows.append(row)
    result = dict(
        schema=profile["schema"], job=job, test_case=test_case, condition=condition, clock=clock,
        cohort=cohort, rows=rows, smoke_only=profile["smoke_only"],
        complete_course_evaluated=not any(r["budget_censored"] for r in rows),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        calibration_id=saved["calibration"]["calibration_id"], model_sha256=saved["model_sha256"],
        train_counters=saved["counters"], information_contract=contract(s),
        evaluation_sensing=s["sensing"], task=c, deployment_parameter_updates=0,
        terminal_value_in_deployment=False,
        nominal_future_hold_approximation=(clock == "irregular"),
        imagined_environment="current causal local scan and material estimate, no future acquired evidence")
    result["diagnostic_version"] = "relative_contact_residual_v1"
    write_json(output, result)
    return result

def gradient_check(profile, job, run_root, artifact_root, device="cpu"):
    s, c, artifact, run = paths(profile, job, run_root, artifact_root)
    saved, _ = initial_saved(profile, dict(job, method="shooting"), s, c, artifact)
    ctl = make_controller(saved, job["seed"], device)
    env = PhysicalEnvironment(c, s, 12345, purpose="evaluation")
    try:
        ctl.act(env.current, env.g, env.hold(), env.base.cur_t)
        # Double precision reduces finite-difference cancellation; contact branches can still switch.
        from pure_cpht.mechanics import Robot
        from pure_cpht.contact import Contact
        from .rollout import Imagination
        ctl.robot = Robot(saved["pure_model"]["geometry"], saved["pure_model"]["parameters"],
                          device=device, dtype=torch.float64)
        ctl.contact = Contact(ctl.robot, saved["pure_model"]["contact"], s["pure"])
        ctl.low, ctl.high = ctl.low.double(), ctl.high.double()
        ctl.engine = Imagination(ctl, c, profile["optimization"], job["horizon"])
        context = {k: v.double() for k, v in ctl.context.items()}
        theta = ctl.mean.detach().double()[None].requires_grad_()
        objective = ctl.engine.rollout(context, parameters=theta).objective.mean()
        derivative = torch.autograd.grad(objective, theta)[0]
        generator = torch.Generator(device=device).manual_seed(97)
        direction = torch.randn(theta.shape, generator=generator, device=device, dtype=torch.float64)
        direction = direction / direction.norm()
        analytic = float((derivative * direction).sum())
        rows = []
        for epsilon in (1e-3, 3e-4, 1e-4, 3e-5):
            with torch.no_grad():
                plus = ctl.engine.rollout(context, parameters=theta + epsilon * direction).objective.mean()
                minus = ctl.engine.rollout(context, parameters=theta - epsilon * direction).objective.mean()
            numerical = float((plus - minus) / (2 * epsilon))
            rows.append(dict(epsilon=epsilon, analytic=analytic, finite_difference=numerical,
                             absolute_error=abs(analytic - numerical),
                             relative_error=abs(analytic - numerical) / max(1e-8, abs(analytic), abs(numerical))))
        result = dict(job=job, finite_gradient=bool(torch.isfinite(derivative).all()),
                      gradient_norm=float(derivative.norm()), directions=rows,
                      best_relative_error=min(x["relative_error"] for x in rows),
                      interpretation="Piecewise derivative; a branch change can invalidate a centered comparison")
        write_json(artifact / f"gradient_check_h{round(1000 * job['horizon'])}.json", result)
        return result
    finally:
        env.close()

