"""Compare paired physical-first evaluations using training seeds as replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
from bench.protocol import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--out',default='reports/physical_first.json')
    p.add_argument('--plot',help='Optional comparison PNG')
    a=p.parse_args();reports=[json.loads(Path(path).read_text()) for path in a.evaluations]
    methods=sorted({r['method'] for r in reports})
    if len(methods)>6:raise ValueError('Select at most six methods')
    contexts={(r['training_control']['case'],r['config']['case'],r['settings']['terrain'],r['config']['contact_profile'],
        r['config']['terrain_elevation'],r['cohort'],r['clock'],r['episodes'],r['train_counters']['decisions']) for r in reports}
    if len(contexts)!=1:raise ValueError('Compare one training/test setup, clock, cohort, and actual training budget')
    seeds=[{r['settings']['seed'] for r in reports if r['method']==m} for m in methods]
    if any(seed!=seeds[0] for seed in seeds):raise ValueError('Compared methods need the same independent training seeds')
    if len({(r['method'],r['settings']['seed']) for r in reports})!=len(reports):raise ValueError('Duplicate method/seed report')
    for seed in seeds[0]:
        paired=[r for r in reports if r['settings']['seed']==seed]
        if len({r['calibration_id'] for r in paired})!=1:raise ValueError('Methods must share calibration data per seed')
        signatures={tuple((row['episode_seed'],row['g'],row['audit_truth']['contact_map_sha256']) for row in r['rows']) for r in paired}
        if len(signatures)!=1:raise ValueError('Evaluation episodes and physical courses are not paired')
    metrics=('success_rate','speed_rmse','relative_speed_rmse','speed_band_fraction','completed_fraction','fall','timeout')
    result={}
    for method in methods:
        selected=[r for r in reports if r['method']==method]
        result[method]={}
        for key in metrics:
            values=[r['path_metrics'][key] for r in selected]
            result[method][key]=dict(mean=float(np.mean(values)),training_seed_count=len(values),
                standard_error=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
                per_seed={str(r['settings']['seed']):v for r,v in zip(selected,values)})
    write_json(a.out,dict(results=result,unit='independent training seed',matched_seed_sets=True))
    if a.plot:
        from fjbench.reporting import plot_comparison
        plot_comparison(reports,a.plot)
    print(f'Wrote paired seed-level comparison: {a.out}')


if __name__=='__main__':main()
