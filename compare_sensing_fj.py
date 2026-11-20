"""Matched information ablations; independent training seeds are replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
from bench.protocol import digest, write_json

METRICS = ('success_rate', 'speed_rmse', 'relative_speed_rmse', 'speed_band_fraction', 'completed_fraction', 'fall', 'timeout')


def estimate(values):
    return dict(mean=float(np.mean(values)), training_seed_count=len(values),
        standard_error=float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)


def compare(reports):
    if not reports or any('information_contract' not in r for r in reports):
        raise ValueError('Select evaluations from the new sensing experiment')
    keys = [(r['information_contract']['mode'], r['method'], r['settings']['seed']) for r in reports]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate information-mode/method/seed report')
    contexts = {(r['training_control']['case'], r['config']['case'], r['settings']['terrain'],
        r['config']['contact_profile'], r['config']['terrain_elevation'], r['cohort'], r['clock'],
        r['episodes'], r['train_counters']['decisions']) for r in reports}
    if len(contexts) != 1:
        raise ValueError('Select one training/test case, terrain, cohort, clock, and decision budget')
    groups = sorted(set((mode, method) for mode, method, _ in keys))
    seed_sets = [{seed for m, a, seed in keys if (m, a)==group} for group in groups]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('Use the same independent training seeds in every selected group')
    for seed in sorted(seed_sets[0]):
        selected = [r for r in reports if r['settings']['seed']==seed]
        signatures = {tuple((row['episode_seed'], row['g'], row['audit_truth']['contact_map_sha256']) for row in r['rows']) for r in selected}
        if len(signatures) != 1:
            raise ValueError('Evaluation courses, tasks, and episode seeds must be paired')
        # The only changed configuration field in a sensor ablation is mode.
        for field in ('settings', 'training_control', 'config', 'evaluation_sensing'):
            values = []
            for r in selected:
                value = json.loads(json.dumps(r[field]))
                if field=='settings':value['sensing'].pop('mode', None)
                if field=='evaluation_sensing':value.pop('mode', None)
                values.append(digest(value))
            if len(set(values)) != 1:
                raise ValueError(f'Unmatched {field}; vary only the information mode')
        for mode in {r['information_contract']['mode'] for r in selected}:
            paired = [r for r in selected if r['information_contract']['mode']==mode]
            if len({r['calibration_id'] for r in paired}) != 1:
                raise ValueError('Methods within each mode must share the calibration artifact')
            if len({digest(r['information_contract']) for r in paired}) != 1:
                raise ValueError('Methods within each mode must share the complete information contract')
    results = {}
    lookup = {key: report for key, report in zip(keys, reports)}
    for mode, method in groups:
        selected = [r for r in reports if r['information_contract']['mode']==mode and r['method']==method]
        results[mode+'/'+method] = dict(metrics={name: estimate([r['path_metrics'][name] for r in selected]) for name in METRICS},
            policy_physics_steps={str(r['settings']['seed']): r['train_counters']['policy_physics_steps'] for r in selected},
            interpretation='same permitted information; model priors, supervision use, controller memory and compute differ by method')
    differences = {}
    seeds = sorted(seed_sets[0])
    for mode in {group[0] for group in groups}:
        if (mode, 'ct_sac') in groups and (mode, 'pure_cpht') in groups:
            differences[mode+'/pure_cpht-minus-ct_sac'] = {name: estimate([
                lookup[(mode, 'pure_cpht', seed)]['path_metrics'][name]-lookup[(mode, 'ct_sac', seed)]['path_metrics'][name]
                for seed in seeds]) for name in METRICS}
    for method in {group[1] for group in groups}:
        for first, second in (('state', 'state_belief'), ('state_belief', 'sensed_belief')):
            if (first, method) in groups and (second, method) in groups:
                differences[method+'/'+second+'-minus-'+first] = {name: estimate([
                    lookup[(second, method, seed)]['path_metrics'][name]-lookup[(first, method, seed)]['path_metrics'][name]
                    for seed in seeds]) for name in METRICS}
    return dict(results=results, paired_differences=differences, unit='independent training seed',
        matched_courses=True, matched_information_within_mode=True,
        uncertainty='standard error across training seeds; one seed gives no standard error')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations', nargs='+', required=True)
    p.add_argument('--out', default='reports/sensing_comparison.json')
    a = p.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in a.evaluations]
    write_json(a.out, compare(reports))
    print('Wrote matched sensing comparison: '+a.out)


if __name__=='__main__':
    main()
