"""Scientific task plots. No metric is replaced by a video-frame statistic."""
from pathlib import Path
import numpy as np


def plot_evaluation(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in result['rows'] if 'path_metrics' in r]
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for row in rows[:3]:
        trace=row['path_metrics']['trajectory']
        t=[x['time'] for x in trace];v=[x['speed'] for x in trace];remaining=[x['remaining'] for x in trace]
        label=f"seed {row['episode_seed']} / {row['path_metrics']['terminal_reason']}"
        line=axes[0].plot(t,v,label=label)[0]
        axes[0].axhline(row['g'],ls='--',lw=1,color=line.get_color(),alpha=.5)
        axes[1].plot(t,remaining,label=label)
    axes[0].set(xlabel='physical time [s]',ylabel='forward COM speed [m/s]')
    axes[0].set_title('Dashed lines: target speed')
    axes[1].set(xlabel='physical time [s]',ylabel='distance to goal [m]')
    axes[1].axhline(0,color='black',lw=.8)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    metrics=result['path_metrics']
    fig.suptitle(f"{result['config']['case']}/{result['config']['experiment']} | {result['method']} | {result['cohort']} / {result['clock']}\ncompletion {metrics['success_rate']:.1%}; speed RMSE {metrics['speed_rmse']:.3f} m/s")
    Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)


def plot_comparison(reports,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not reports:raise ValueError('No evaluation files')
    contexts={(r['config']['case'],r['config']['experiment'],r['cohort'],r['clock'],r['split']) for r in reports}
    if len(contexts)!=1:raise ValueError('Compare one case, experiment, cohort, split, and clock at a time')
    methods=sorted({r['method'] for r in reports})
    seeds=[frozenset(r['config']['seed'] for r in reports if r['method']==method) for method in methods]
    if len(set(seeds))>1:raise ValueError('Use the same independent training seeds for each compared method')
    from .config import calibration_key
    signatures={(calibration_key(dict(r['config'],seed=0)),r['config']['total_decisions']) for r in reports}
    if len(signatures)>1:raise ValueError('Compared evaluations use different physical, task, measurement, preparation, or budget specifications')
    if len(methods)>6:raise ValueError('Select at most six methods for a readable comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,('success_rate','speed_rmse'),('completion rate','speed RMSE [m/s]')):
        for i,method in enumerate(methods):
            values=[r['path_metrics'][key] for r in reports if r['method']==method]
            ax.bar(i,np.mean(values),alpha=.55)
            ax.scatter(np.full(len(values),i),values,color='black',s=18)
        ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right');ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0,1.05)
    fig.suptitle('Each dot is one independent training seed; task strata averaged first')
    Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)
