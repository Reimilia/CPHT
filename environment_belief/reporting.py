"""Read-only diagnostic plots; truth is confined to evaluation output."""
from pathlib import Path
import numpy as np


def plot_trace(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        return
    t = np.asarray([r['time'] for r in rows])
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for ax, name, truth_key, std_key, label in zip(axes[:4],
            ('height', 'slope', 'curvature', 'friction'), ('height', 'slope', 'curvature', 'local_patch_friction'),
            ('height_std', 'slope_std', 'curvature_std', 'friction_std'), ('elevation [m]', 'slope', 'curvature [1/m]', 'friction')):
        estimate = np.asarray([r['estimate'][name] for r in rows])
        ax.plot(t, [r['truth'][truth_key] for r in rows], color='black', label='audit reference')
        ax.plot(t, estimate, color='#147d92', label='causal estimate')
        if std_key:
            std = np.asarray([r['estimate'][std_key] for r in rows])
            ax.fill_between(t, np.maximum(0, estimate-2*std), estimate+2*std, alpha=.2, color='#147d92', label='heuristic +/-2 std')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(loc='best')
    axes[3].legend(loc='best')
    axes[4].plot(t, [r['estimate']['geometry_sensor_age'] if r['estimate']['geometry_sensor_valid'] else np.nan for r in rows], label='geometry sensor age')
    axes[4].plot(t, [r['estimate']['friction_age'] if r['estimate']['friction_valid'] else np.nan for r in rows], label='friction evidence age')
    axes[4].set(xlabel='physical time [s]', ylabel='age [s]')
    axes[4].legend(loc='best')
    for event in rows[-1].get('friction_events', []):
        for ax in axes:
            ax.axvline(event['applied_time'], color='#b65339', alpha=.4, linestyle='--', linewidth=1.)
    fig.suptitle('Environment assimilation | friction truth is the patch below the torso')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
