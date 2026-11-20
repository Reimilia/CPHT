"""Select a checkpoint using validation completion and speed metrics only."""
import hashlib
import json
from pathlib import Path
from fjbench.cli import parser,resolve
from fjbench.checkpoints import checkpoint_path
from bench.protocol import write_json


def choose(directory,clocks,episodes):
    directory=Path(directory);candidates={}
    for clock in clocks:
        for path in (directory/'evaluations').glob(f'*_validation_iid_{clock}_n{episodes}.json'):
            value=json.loads(path.read_text())
            label=value['checkpoint']
            if label=='selected':continue
            if value['split']!='validation' or value['cohort']!='iid' or value['clock']!=clock:
                raise ValueError('Validation filename and content differ')
            checkpoint=checkpoint_path(directory,label)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=value['checkpoint_sha256']:
                raise ValueError('Stale validation result for '+str(checkpoint))
            candidates.setdefault(label,{})[clock]=value
    available=[]
    for label,values in candidates.items():
        if set(values)!=set(clocks):continue
        first=values[clocks[0]]
        if any(v['config']!=first['config'] or v['method']!=first['method'] or v['checkpoint_sha256']!=first['checkpoint_sha256'] for v in values.values()):
            raise ValueError('Validation clocks use different checkpoints or configurations')
        if first.get('path_metrics'):
            completion=sum(v['path_metrics']['success_rate'] for v in values.values())/len(clocks)
            error=sum(v['path_metrics']['relative_speed_rmse'] for v in values.values())/len(clocks)
            key=(completion,-error)
        else:key=(sum(v['family_time_score'] for v in values.values())/len(clocks),0.)
        available.append((key,label,first))
    if not available:raise FileNotFoundError('No checkpoint has validation results for every requested clock')
    _,label,value=max(available,key=lambda x:(x[0],x[1]))
    path=checkpoint_path(directory,label)
    return dict(checkpoint=label,checkpoint_file=path.name,checkpoint_sha256=value['checkpoint_sha256'],
        method=value['method'],config=value['config'],validation_clocks=clocks,validation_episodes=episodes,
        criterion='completion rate, then relative speed RMSE' if value.get('path_metrics') else 'family time score',
        candidates=sorted(x[1] for x in available),test_results_used=False)


def main():
    p=parser(__doc__,True)
    p.add_argument('--clocks',default='irregular,fixed');p.add_argument('--episodes',type=int,default=20)
    a=p.parse_args();c,method,_,directory=resolve(a)
    clocks=a.clocks.split(',')
    if a.episodes<1 or any(x not in ('irregular','fixed') for x in clocks):p.error('Invalid validation clock/count')
    if a.dry_run:
        print(json.dumps(dict(stage='select',run=str(directory),clocks=clocks,episodes=a.episodes,
            uses_split='validation',expensive_work_launched=False),indent=2));return
    chosen=choose(directory,clocks,a.episodes)
    if chosen['config']!=c or chosen['method']!=method:raise ValueError('Validation results do not match this job')
    write_json(directory/'selected_checkpoint.json',chosen)
    print(json.dumps(chosen,indent=2))


if __name__=='__main__':main()
