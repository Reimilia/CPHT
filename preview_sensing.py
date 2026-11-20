"""Short sensor/assimilation precheck with optional headless video; no training."""
import json
import os
from pathlib import Path
from physical_fj.cli import parser, resolve


def main():
    p = parser(__doc__)
    p.set_defaults(options='cases/sensing.json', information='sensed_belief')
    p.add_argument('--seconds', type=float, default=.5)
    p.add_argument('--out', default='prechecks/sensing')
    p.add_argument('--video', action='store_true')
    p.add_argument('--random-actions', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--gl', choices=['egl', 'osmesa'], default='egl')
    a = p.parse_args()
    s, c, _, _ = resolve(a)
    if not 0 < a.seconds <= c['horizon']:
        p.error('Precheck duration must be positive and within the task horizon')
    if a.dry_run:
        print(json.dumps(dict(stage='sensing_precheck', mode=s['sensing']['mode'], seconds=a.seconds,
            video=a.video, training_updates=0, expensive_work_launched=False), indent=2))
        return
    os.environ['MUJOCO_GL'] = a.gl
    os.environ['PYOPENGL_PLATFORM'] = a.gl
    import numpy as np
    from bench.protocol import write_json
    from physical_fj.mechanics import PhysicalEnvironment
    from environment_belief.runtime import summarize
    from environment_belief.reporting import plot_trace
    from fjbench.recording import EpisodeRecorder
    env = PhysicalEnvironment(c, s, 100000+a.seed, purpose='evaluation')
    rng = np.random.default_rng(a.seed+902)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recorder = None
    rows = [env.sensing.audit()]
    try:
        if a.video:
            recorder = EpisodeRecorder(env, out/'sensing.mp4', dict(width=640, height=480, fps=25,
                seconds=a.seconds, end_card_seconds=0., overwrite=a.overwrite),
                'Sensor precheck | '+s['sensing']['mode'])
        while env.base.cur_t < a.seconds:
            action = rng.uniform(-1, 1, 6) if a.random_actions else np.zeros(6)
            _, info, _ = env.step(action)
            rows.append(env.sensing.audit())
            if info['terminated'] or info['truncated']:
                break
        report = dict(settings=s, control=c, information_contract=env.current['information_contract'],
            observation_dimension=len(env.current['raw']), assimilation=summarize(rows), trace=rows,
            path_metrics=env.path_task.summary(), training_updates=0,
            scope='sensor plumbing precheck with zero/random actions; locomotion and parameter accuracy remain unvalidated')
        if recorder:
            recorder.capture(env.base.cur_t)
            report['video'] = recorder.close(True)
            recorder = None
        write_json(out/'sensing.json', report)
        plot_trace(rows, out/'sensing.png')
        print(json.dumps(dict(output=str(out), information=s['sensing']['mode'],
            dimension=report['observation_dimension'], sensor=report['assimilation']['sensor'], training_updates=0)))
    finally:
        if recorder:
            recorder.close(False)
        env.close()


if __name__ == '__main__':
    main()
