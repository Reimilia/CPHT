"""Patch finite recovery in an existing path-aware project; preserve its config."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();source=Path(__file__).resolve().parent;target=Path(a.target).resolve()
    if target==source:p.error('Use this updated project directly')
    if not (target/'fjbench/finite.py').exists() or not (target/'fjbench/agent.py').exists():p.error('Target must be an existing F/J project root')
    files=('fjbench/finite.py','fjbench/agent.py','diagnose_finite_fj.py','docs/PHYSICAL_FIRST.md')
    if a.dry_run:
        for rel in files:print(target/rel)
        return
    backup=target/'finite_fix_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for rel in files:
        destination=target/rel
        if destination.exists():
            copy=backup/rel;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,copy)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,destination)
    print(f'Installed finite recovery patch. Backups: {backup}')
    print('The existing configuration and prepared data remain in use. The new physical-first pipeline is in the full bundle.')


if __name__=='__main__':main()
