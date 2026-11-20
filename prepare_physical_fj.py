"""Calibrate positive physical pH coefficients; fit no observer or value operator."""
import json
import os
from physical_fj.cli import parser,resolve


def main():
    p=parser(__doc__);a=p.parse_args();s,c,artifact,_=resolve(a)
    if a.dry_run:
        print(json.dumps(dict(stage='physical_calibration',settings=s,output=str(artifact),fitted_parameters=3,
            measurement='instrumented state, acceleration and force ports',value_operator_used=False,expensive_work_launched=False),indent=2));return
    os.environ.setdefault('MUJOCO_GL','egl')
    from physical_fj.prepare import prepare
    report=prepare(s,c,artifact)
    print(json.dumps(dict(artifact=str(artifact),coefficients=report['coefficients'],validation=report['metrics']['validation'],policy_updates=0)))
    if 'pure' in s:
        from pure_cpht.learning import prepare_artifact
        pure=prepare_artifact(artifact,s)
        print(json.dumps(dict(method='pure_cpht',mechanical_parameters=32,contact_parameters=4,
            metrics=pure['metrics'],actor_updates=0,critic_updates=0)))


if __name__=='__main__':main()
