"""Train a selected measured-transition SAC/CT-SAC physical-feature ablation."""
import json
import os
from physical_fj.cli import parser,resolve


def main():
    p=parser(__doc__,True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    s,c,artifact,directory=resolve(a)
    if a.dry_run:
        print(json.dumps(dict(stage='physical_first_train',method=a.method,settings=s,calibration=str(artifact),
            run=str(directory),value_inverse_solves=0,expensive_work_launched=False),indent=2));return
    if not (artifact/'calibration.json').exists():p.error('Run prepare_physical_fj.py with the same settings first')
    if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text()).get('complete'):
        print(f'Completed run: {directory}');return
    os.environ.setdefault('MUJOCO_GL','egl')
    if a.method=='pure_cpht':
        from pure_cpht.training import train
    else:
        from physical_fj.train import train
    print(json.dumps(train(s,c,a.method,artifact,directory,a.device,a.resume)))


if __name__=='__main__':main()
