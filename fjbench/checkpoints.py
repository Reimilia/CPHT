"""Checkpoint labels shared by evaluation, selection, and aggregation."""
import hashlib
import json
from pathlib import Path


def checkpoint_path(directory,label):
    directory=Path(directory)
    if label=='selected':
        chosen=json.loads((directory/'selected_checkpoint.json').read_text())
        filename=chosen['checkpoint_file']
        if Path(filename).name!=filename:raise ValueError('Selected checkpoint must be inside its run directory')
        path=directory/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=chosen['checkpoint_sha256']:
            raise ValueError('Selected checkpoint content changed; repeat validation selection')
        return path
    if label in ('final','latest'):return directory/(label+'.pt')
    if label.isdecimal():return directory/f'policy_{int(label):09d}.pt'
    if label.startswith('policy_') and label[7:].isdecimal():return directory/(label+'.pt')
    raise ValueError('Use final, latest, a decision count, or selected')
