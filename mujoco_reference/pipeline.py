"""Paired F/J evaluation; no fitting and no actor/critic updates."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
from physical_fj.mechanics import PhysicalEnvironment
from pure_cpht.controller import Controller
from .common import read_checkpoint, write, plain
from .controller import NativeController
from .transition import capture, validate

METHODS = ('learned_feedback', 'mujoco_ls', 'mujoco_fd_ls', 'learned_mpc', 'mujoco_mpc')
ACCESS = dict(learned_feedback='causal state/force history, inferred terrain and friction, calibrated pH mechanics',
              learned_mpc='same causal input; rollout uses calibrated pH/contact model and inferred scan',
              mujoco_ls='privileged current MuJoCo mechanics and contact solve; same inferred scan in acceleration target',
              mujoco_fd_ls='privileged current MuJoCo mechanics/contact response under perturbed controls; same inferred scan',
              mujoco_mpc='privileged full compiled terrain/current friction and mechanics throughout prediction; same inferred scan in target/cost')


def checkpoint(root, case):
    paths = sorted((Path(root)/case/'terrain'/'pure_cpht').glob('*/final.pt'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {case}/terrain/pure_cpht/*/final.pt under {root}; found {len(paths)}')
    return paths[0]


def planning_options(saved, args):
    out = dict(saved['settings']['pure'])
    elites = min(3, args.candidates-1) if args.elites is None else args.elites
    out.update(candidates=args.candidates, elites=elites, iterations=args.iterations,
               horizon_steps=args.horizon_steps, replan_seconds=args.replan_seconds)
    return out


def make_controller(env, saved, method, options, seed):
    if method.startswith('mujoco_'):
        return NativeController(env, saved, method, options, seed)
    return Controller(saved['pure_model'], options, saved['neutral'], seed,
                      gait=saved['gait'], feedback_only=method == 'learned_feedback')


def episode(saved, method, options, case, clock, seed, episode_index, output, args):
    env = PhysicalEnvironment(saved['control'], saved['settings'], seed, episode_index,
                              args.cohort, clock, purpose='evaluation', start_x=args.start_x)
    ctrl = make_controller(env, saved, method, options, seed)
    recorder, trace, reward, decisions = None, [], 0., 0
    started = time.perf_counter()
    saturated = projected = target_residual = nonlinear = 0.
    try:
        if args.video_episodes > episode_index:
            from fjbench.recording import EpisodeRecorder
            recorder = EpisodeRecorder(env, output/'videos'/f'{case}_{clock}_{method}_{seed}.mp4',
                        dict(width=640, height=480, fps=25, overwrite=True, end_card_seconds=1.),
                        f'{case} | {method} | '+('MuJoCo privileged reference' if method.startswith('mujoco_') else 'frozen learned model'))
        while True:
            hold, now = env.hold(), env.base.cur_t
            action = ctrl.act(env.current, env.g, hold, now)
            _, info, _ = env.step(action)
            decisions += 1
            reward += info['R']
            d = ctrl.diagnostics
            saturated += info['h']*float(np.mean(abs(action) > .99))
            projected += info['h']*float(d.get('ls_projected_gradient_inf', d.get('feedback_projected_residual', 0.)))
            target_residual += info['h']*float(d.get('weighted_acceleration_residual', 0.))
            nonlinear += info['h']*float(d.get('contact_linearization_error', 0.))
            trace.append(dict(time=float(env.base.cur_t), h=info['h'], action=action,
                              q=env.current['raw'][:9], v=env.current['raw'][9:18],
                              progress=info['path_metrics']['completed_fraction'],
                              com_speed=env.path_task.last_v, phase=ctrl.phase,
                              plans=ctrl.plans, saturation=float(np.mean(abs(action) > .99)),
                              weighted_acceleration_residual=d.get('weighted_acceleration_residual'),
                              contact_linearization_error=d.get('contact_linearization_error')))
            if info['terminated'] or info['truncated']:
                break
            if args.max_seconds is not None and env.base.cur_t >= args.max_seconds-1e-9:
                break
        elapsed = env.path_task.elapsed
        metrics = env.path_task.summary()
        if not info['terminated'] and not info['truncated']:
            metrics['terminal_reason'] = 'diagnostic_cutoff'
        row = dict(case=case, clock=clock, method=method, seed=seed, episode=episode_index,
                   path_metrics=metrics, wall_seconds=time.perf_counter()-started, decisions=decisions,
                   saturation_time_mean=saturated/elapsed, projected_residual_time_mean=projected/elapsed,
                   weighted_acceleration_residual_time_mean=target_residual/elapsed if method.startswith('mujoco_') else None,
                   contact_linearization_error_time_mean=nonlinear/elapsed if method.startswith('mujoco_') else None,
                   predictive_steps=ctrl.predicted_steps, control_optimizations=ctrl.plans,
                   native_forward_calls=getattr(ctrl, 'forward_calls', 0),
                   checkpoint_sha256=saved['_sha256'], information_access=ACCESS[method],
                   task_reward_sum=reward, deployment_model_refits=0, actor_updates=0, critic_updates=0,
                   audit_truth=env.audit_truth(), trace_file=f'traces/{case}_{clock}_{method}_{seed}.json')
        if recorder:
            recorder.capture(env.base.cur_t)
            row['video'] = recorder.close(True, row)
            recorder = None
        write(output/row['trace_file'], trace)
        return row
    finally:
        if recorder:
            recorder.close(False)
        env.close()


def audit_upload(root, output):
    report = dict(source='supplied sensor.zip', runs=[], ct_sac_checkpoints=[])
    for path in sorted(Path(root).rglob('run.json')):
        run = json.loads(path.read_text())
        episodes_path = path.parent/'episodes.jsonl'
        lines = [json.loads(s) for s in episodes_path.read_text().splitlines() if s.strip()] if episodes_path.exists() else []
        rows = [r.get('path_metrics', r) for r in lines]
        complete = [r for r in rows if r.get('terminal_reason') in ('goal', 'fall', 'timeout')]
        recent = complete[-20:]
        sensing = run.get('settings', {}).get('sensing', {})
        entry = dict(path=str(path.relative_to(root)), sensing_mode=sensing.get('mode'),
                     external_geometry_sensor=sensing.get('mode') == 'sensed_belief',
                     checkpoint_present=(path.parent/'final.pt').exists(),
                     completed_training_episodes=len(complete), recent_completed_training_episodes=len(recent),
                     recent_terminal_counts={k:sum(r.get('terminal_reason') == k for r in recent) for k in ('goal', 'fall', 'timeout')},
                     recent_mean_completed_fraction=float(np.mean([r['completed_fraction'] for r in recent])) if recent else None,
                     interpretation='recent completed training episodes; not independent frozen-policy evaluation')
        failure = path.parent/'model_refit_failure.json'
        if failure.exists():
            entry['refit_failure'] = json.loads(failure.read_text())
        report['runs'].append(entry)
    report['ct_sac_checkpoints'] = [str(p.relative_to(root)) for p in Path(root).rglob('final.pt') if 'ct_sac' in p.parts]
    write(output/'upload_audit.json', report)
    return report


def report(results, output):
    lines = ['# MuJoCo reference comparison', '',
             'No training or parameter refitting was run. Each row is one paired episode. '
             'F fixes the physical regime/course; J draws the regime/course per episode. '
             'These runs do not impose an additional within-episode environment change.', '',
             '| Case | Clock | Method | End | Distance (m) | Progress | Speed RMSE (m/s) | Saturation | Wall (s) |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in results['rows']:
        p = r['path_metrics']
        lines.append(f"| {r['case']} | {r['clock']} | {r['method']} | {p['terminal_reason']} | {p['distance_travelled']:.3f} | {p['completed_fraction']:.1%} | {p['speed_rmse']:.3f} | {r['saturation_time_mean']:.1%} | {r['wall_seconds']:.1f} |")
    lines += ['', 'The MuJoCo arms use privileged model access. Their outcomes diagnose the selected controller and its finite gait/search space. '
              'A failure is not a proof of terrain infeasibility. Learned MPC and MuJoCo MPC share saved initial gait, parameter bounds, '
              'search count, cost weights, and control horizon; they also differ in feedback mechanics, contact model, numerical step, '
              'and scan-coverage cost (native: torso; learned: capsule points). This is a practical reference comparison, not a one-factor model ablation.', '',
              'Training counters remain unchanged. Wall times include Python feedback and observation overhead and are not real-time guarantees.', '']
    (output/'report.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cases = list(dict.fromkeys(r['case'] for r in results['rows']))
        fig, axes = plt.subplots(len(cases), 2, figsize=(11, 3.4*len(cases)), squeeze=False)
        for row in results['rows']:
            if row['clock'] != results['clocks'][0] or row['episode'] != 0:
                continue
            i = cases.index(row['case'])
            data = row['path_metrics']['trajectory']
            axes[i, 0].plot([d['time'] for d in data], [d['x']-row['path_metrics']['initial_x'] for d in data], label=row['method'])
            axes[i, 1].plot([d['time'] for d in data], [d['speed'] for d in data], label=row['method'])
        for i, case in enumerate(cases):
            axes[i, 0].axhline(20., color='black', ls=':', lw=1)
            axes[i, 1].axhline(3., color='black', ls=':', lw=1)
            for j, label in enumerate(('Distance (m)', 'COM speed (m/s)')):
                axes[i, j].set(xlabel='Physical time (s)', ylabel=label, title=f'{case}: {results["clocks"][0]}, first paired episode')
                axes[i, j].grid(alpha=.2)
            axes[i, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'comparison.png', dpi=160)
        plt.close(fig)
    except ImportError:
        pass


def run(args):
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit_upload(Path(args.sensor_root), output)
    result = dict(schema='mujoco_reference_v1', methods=args.methods, cases=args.cases, clocks=args.clocks,
                  episodes_per_cell=args.episodes, initial_seed=args.seed, parameters=vars(args),
                  privileged_access=ACCESS, training_launched=False, model_refits=0,
                  mujoco_version=mujoco.__version__, rows=[])
    for case in args.cases:
        saved = read_checkpoint(checkpoint(args.sensor_root, case))
        options = planning_options(saved, args)
        for clock in args.clocks:
            for i in range(args.episodes):
                for method in args.methods:
                    row = episode(saved, method, options, case, clock, args.seed+i, i, output, args)
                    result['rows'].append(row)
                    write(output/'results.json', result)
                    p = row['path_metrics']
                    print(f"{case} {clock} {method}: {p['terminal_reason']}, distance={p['distance_travelled']:.3f} m, speed_rmse={p['speed_rmse']:.3f}, wall={row['wall_seconds']:.1f} s", flush=True)
    report(result, output)
    return result


def samples(args):
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in args.cases:
        saved = read_checkpoint(checkpoint(args.sensor_root, case))
        # These are local terrain stress snapshots, independent of the full
        # course evaluation and explicitly labeled by their reset location.
        env = PhysicalEnvironment(saved['control'], saved['settings'], args.seed, clock=args.clock,
                                  purpose='evaluation', start_x=args.sample_start_x)
        ctrl = NativeController(env, saved, 'mujoco_fd_ls', saved['settings']['pure'], args.seed)
        try:
            while env.base.cur_t < args.warmup_seconds-1e-9:
                action = ctrl.act(env.current, env.g, env.hold(), env.base.cur_t)
                _, info, _ = env.step(action)
                if info['terminated'] or info['truncated']:
                    raise RuntimeError('Local diagnostic pose terminated during warmup; lower --warmup-seconds')
            path = output/f'{case}_transition.json'
            sample = capture(env, ctrl, saved, path)
            results.append(dict(case=case, checks=sample['checks'], standalone_validation=validate(path)))
            print(json.dumps(plain(results[-1])), flush=True)
        finally:
            env.close()
    write(output/'validation.json', results)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('compare', 'sample'):
        s = sub.add_parser(name)
        s.add_argument('--sensor-root', required=True, help='Extracted sensor directory containing F/ and J/')
        s.add_argument('--output', required=True)
        s.add_argument('--cases', nargs='+', choices=['F', 'J'], default=['F', 'J'])
        s.add_argument('--seed', type=int, default=100000)
        s.add_argument('--threads', type=int, default=1)
    c = sub.choices['compare']
    c.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    c.add_argument('--clocks', nargs='+', choices=['fixed', 'irregular'], default=['fixed', 'irregular'])
    c.add_argument('--episodes', type=int, default=1)
    c.add_argument('--cohort', choices=['iid', 'ood'], default='iid')
    c.add_argument('--candidates', type=int, default=8)
    c.add_argument('--elites', type=int, help='Default: min(3, candidates-1)')
    c.add_argument('--iterations', type=int, default=1)
    c.add_argument('--horizon-steps', type=int, default=20)
    c.add_argument('--replan-seconds', type=float, default=.1)
    c.add_argument('--max-seconds', type=float)
    c.add_argument('--start-x', type=float, default=0.)
    c.add_argument('--video-episodes', type=int, default=0)
    s = sub.choices['sample']
    s.add_argument('--clock', choices=['fixed', 'irregular'], default='fixed')
    s.add_argument('--sample-start-x', type=float, default=6.)
    s.add_argument('--warmup-seconds', type=float, default=.2)
    v = sub.add_parser('validate')
    v.add_argument('transition')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == 'validate':
        print(json.dumps(validate(args.transition), indent=2))
    else:
        if args.threads < 1:
            p.error('--threads must be positive')
        if args.command == 'compare':
            if args.candidates < 2 or min(args.iterations, args.horizon_steps, args.episodes) < 1 or args.replan_seconds <= 0:
                p.error('Use candidates>=2 and positive counts/replan time')
            if args.elites is not None and not 1 <= args.elites < args.candidates:
                p.error('--elites must lie between 1 and candidates-1')
            if args.max_seconds is not None and args.max_seconds <= 0:
                p.error('--max-seconds must be positive')
            run(args)
        else:
            if args.warmup_seconds < 0:
                p.error('--warmup-seconds cannot be negative')
            samples(args)
