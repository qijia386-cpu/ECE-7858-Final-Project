"""
Experiment 1 -- replicate Fig. 3A (pattern evolution).

Paper Fig. 3A shows snapshots at t=0, 10, 400 for a=0.5, b=9.6 and claims
that the t=400 panel "is very similar to t=10".  In our re-implementation
the patches state forms within ~10 steps and at b/a slightly above the
critical line (b/a >= 20) survives with limited drift over hundreds of
steps; below the critical line the system slowly coarsens further.

To make the comparison transparent we render a *four-panel* row:

    t = 0    -- random initial state
    t = 10   -- patch nucleation (paper's left/middle panels)
    t = 50   -- mature patches state (where alpha ~ 1, paper's claim)
    t = 400  -- long time (still many patches, with some coarsening)

Two parameter choices are shown side by side:
    (i)  paper reference  a=0.5, b=9.6  (b/a = 19.2, just below critical)
    (ii) on critical line a=0.5, b=10.0 (b/a = 20.0)
to demonstrate that the patches state is more stable on/above the
critical line, consistent with the paper's phase diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from simulation import run_simulation
from analysis import patch_sizes, correlation_length


def _run_one(a, b, seed, snap_times):
    return run_simulation(
        n_steps=max(snap_times),
        a=a, b=b,
        rule='maximum',
        neighborhood='euclidean',
        include_self=False,
        f_random=0.05,
        snapshot_steps=snap_times,
        seed=seed,
    )


def run(seed=2024, save_path='fig3A_snapshots.png'):
    snap_times = [0, 10, 50, 400]
    # (i) paper reference and (ii) on critical line
    runs = [
        ('a=0.5, b=9.6  (b/a=19.2, paper Fig. 3 reference)', 0.5, 9.6),
        ('a=0.5, b=10.0 (b/a=20.0, critical line)',          0.5, 10.0),
    ]
    results = [(label, _run_one(a, b, seed, snap_times)) for label, a, b in runs]

    cmap = ListedColormap(['#2ca02c', '#d62728', '#1f77b4', '#ffdd33'])
    n_rows = len(results)
    n_cols = len(snap_times)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.7 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    short_labels = [
        r'$a=0.5,\ b=9.6$' + '\n' + r'$(b/a=19.2)$',
        r'$a=0.5,\ b=10.0$' + '\n' + r'$(b/a=20.0)$',
    ]

    for r, (label, res) in enumerate(results):
        for c, t in enumerate(snap_times):
            ax = axes[r, c]
            state = res['snapshots'][t]
            ax.imshow(state, cmap=cmap, vmin=-0.5, vmax=3.5,
                      interpolation='nearest')
            ps = patch_sizes(state)
            eps = correlation_length(state)
            n_patches = len(ps)
            max_patch = int(ps.max()) if ps.size > 0 else 0
            H = res['harvest_trajectory'][t]
            if r == 0:
                ax.set_title(f't = {t}', fontsize=12)
            ax.set_xlabel(
                f'H={H:.2f}, eps={eps:.1f}\n'
                f'#patches={n_patches}, max={max_patch}',
                fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([])
        # short row label on the leftmost panel (no overlap)
        axes[r, 0].set_ylabel(short_labels[r], fontsize=11, labelpad=10)

    fig.suptitle(
        'Fig. 3A replication: pattern evolution under the maximum-harvest '
        'rule\n(top: paper reference; bottom: critical line)',
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')

    for label, res in results:
        H = res['harvest_trajectory']
        print(f'  [{label}]')
        for t in snap_times:
            print(f'    H(t={t:>3d}) = {H[t]:.3f}')
    return results


if __name__ == '__main__':
    run()
