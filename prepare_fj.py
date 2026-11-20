"""Collect and fit a shared measurement/operator artifact for a case and seed."""
import json
import os
from fjbench.cli import parser,resolve


def main():
    p=parser(__doc__);a=p.parse_args();c,_,prepared,_=resolve(a)
    if a.dry_run:
        print(json.dumps(dict(stage="prepare",config=c,output=str(prepared),
            expensive_work_launched=False,observer_shared_by_all_methods=True),indent=2));return
    if prepared.exists():
        print(f"Reusing common calibration: {prepared}");return
    if a.device=="cpu":os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.setdefault("MUJOCO_GL","egl")
    from fjbench.calibration import prepare
    prepared.parent.mkdir(parents=True,exist_ok=True)
    lock=prepared.parent/"prepare.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    try:prepare(c,prepared.parent,a.device)
    finally:lock.unlink(missing_ok=True)
    print(f"Prepared common measured-data artifact: {prepared}")


if __name__=="__main__":main()
