"""Train a case F/J policy with its common measurement interface and operator."""
import json
import os
import traceback
from bench.protocol import write_json
from fjbench.cli import parser,resolve
from fjbench.config import label


def main():
    p=parser(__doc__,True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--pause-hours",type=float)
    a=p.parse_args();c,method,prepared,directory=resolve(a)
    if a.dry_run:
        print(json.dumps(dict(stage="train",config=c,method=method,label=label(c,method),
            calibration=str(prepared),run=str(directory),
            policy_decision_budget=c["total_decisions"]-c["calibration_decisions"],
            calibration_decisions_charged=c["calibration_decisions"],expensive_work_launched=False),indent=2));return
    if not prepared.exists():p.error("Prepare the shared artifact first with prepare_fj.py and the same case, experiment, seed, and settings")
    status=directory/"status.json"
    if status.exists() and json.loads(status.read_text())["status"]=="complete":
        print(f"Already complete: {directory}");return
    if directory.exists() and any(directory.iterdir()) and not a.resume:
        p.error("Run contains an attempt. Use --resume or a new root.")
    directory.mkdir(parents=True,exist_ok=True)
    lock=directory/"run.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    if a.device=="cpu":os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.setdefault("MUJOCO_GL","egl")
    try:
        from fjbench.runner import train
        result=train(c,method,prepared,directory,a.device,a.resume,a.pause_hours)
    except Exception as e:
        write_json(status,dict(status="failed",error=str(e),traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)
    print(json.dumps(dict(run=str(directory),**result),indent=2))


if __name__=="__main__":main()
