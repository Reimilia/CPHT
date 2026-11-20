"""Install the coverage fix in an existing path-aware F/J project, with backups."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

FILES=('fjbench/config.py','fjbench/data.py','fjbench/calibration.py',
       'check_calibration_fj.py','docs/history/README_CALIBRATION_FIX.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,help='Existing project folder, including a folder named TPHT_CT_RL_Benchmark_v2')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();source=Path(__file__).resolve().parent;target=Path(a.target).resolve()
    if target==source:p.error('This is already the patched project; use its preparation script directly')
    environment=target/'fjbench/environment.py';models=target/'fjbench/models.py'
    if not environment.exists() or not models.exists():p.error('Target must be an existing F/J project root')
    if not all(word in environment.read_text() for word in ('purpose','initial_speed','path_task')) or 'with_survival' not in models.read_text():
        p.error('This patch requires the path-aware v3.1/v3.2 F/J code; use the full updated bundle for an earlier project')
    if any(not (source/rel).is_file() for rel in FILES):p.error('Extract the full fixed bundle before running this installer')
    if a.dry_run:
        print(json.dumps(dict(target=str(target),files=FILES,applied=False),indent=2));return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=target/'calibration_fix_backups'/stamp
    for rel in FILES:
        destination=target/rel
        if destination.exists():
            copy=backup/rel;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,copy)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,destination)
    metadata=dict(patch='goal_balanced_v2',applied_utc=stamp,backup=str(backup),
        files={rel:hashlib.sha256((target/rel).read_bytes()).hexdigest() for rel in FILES})
    (target/'CALIBRATION_PATCH_VERSION.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Installed calibration fix in {target}. Backups: {backup}')
    print('Run STAGE=prepare DRY_RUN=0 bash scripts/run_path_speed.sh from that project.')


if __name__=='__main__':main()
