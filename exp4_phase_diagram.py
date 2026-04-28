"""
Experiment 4 -- replicate Fig. 4.

Sweep over (a, b) in [0.04, 0.5] x [0.5, 10.0] (paper's range) and at each
point measure:
  (A) average harvest H,
  (B) cumulative-CCDF tail exponent alpha,
  (C) correlation length epsilon (Eq. 2).

To reduce single-snapshot noise, each cell is averaged over `n_replicates`
random seeds and over a small window of measurement times within the
patches-state attractor (t in measure_times). The total simulation length
is set to the largest measure_time.

The paper's main claim is that the critical region -- power-law patches
with long correlation length -- runs along the diagonal b/a ~ 20.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from simulation import run_simulation
from analysis import patch_sizes, fit_powerlaw_clauset, correlation_length


# Default scan ranges (paper's Fig. 4 spans 0 < a < 0.5, 0 < b < 10).
# Match the paper's resolution by using a 25 x 25 grid (paper Fig. 4 looks
# like ~30 a-bins by ~25 b-bins).  This is ~4x more cells than the original
# 12x12 grid and gives a smoother phase diagram.
DEFAULT_A = np.linspace(0.02, 0.5, 25)
DEFAULT_B = np.linspace(0.4, 10.0, 25)
# Measure when the patches state is mature.  Paper claims that t=400 is
# similar to t=10; in our implementation patches mature by t~30-50, so we
# average across a window in that range to suppress single-snapshot noise.
DEFAULT_TIMES = (30, 40, 50, 60)


def sweep(a_values=None, b_values=None,
          measure_times=DEFAULT_TIMES,
          n_replicates=4, seed=2024,
          cache_path='phase_diagram_state.npz',
          time_budget=None):
    """Run the (a, b) sweep, optionally with caching+resume.

    Parameters
    ----------
    measure_times : iterable of int
        Time steps within the patches-state window where measurements are
        taken. Quantities are averaged across these snapshots.
    n_replicates : int
        Number of random-seed replicates per (a, b) cell.
    cache_path : str or None
        If not None, intermediate results are saved to this NPZ. Allows
        resuming after a timeout. Pass None to disable caching.
    time_budget : float or None
        If set, the function returns once this many seconds have elapsed
        even if not all cells are done. The cache lets a subsequent call
        pick up where this one left off.
    """
    a_values = np.asarray(DEFAULT_A if a_values is None else a_values)
    b_values = np.asarray(DEFAULT_B if b_values is None else b_values)
    measure_times = list(measure_times)

    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        # validate that cache matches current grid
        if (cache['a'].shape == a_values.shape
                and np.allclose(cache['a'], a_values)
                and cache['b'].shape == b_values.shape
                and np.allclose(cache['b'], b_values)):
            H_grid = cache['H'].copy()
            alpha_grid = cache['alpha'].copy()
            eps_grid = cache['eps'].copy()
            done = cache['done'].copy()
            print(f'Resuming: {int(done.sum())}/{done.size} cells already done')
        else:
            cache_path = None  # mismatched cache, don't write to it
            H_grid = np.zeros((len(b_values), len(a_values)))
            alpha_grid = np.full((len(b_values), len(a_values)), np.nan)
            eps_grid = np.zeros((len(b_values), len(a_values)))
            done = np.zeros((len(b_values), len(a_values)), dtype=bool)
    else:
        H_grid = np.zeros((len(b_values), len(a_values)))
        alpha_grid = np.full((len(b_values), len(a_values)), np.nan)
        eps_grid = np.zeros((len(b_values), len(a_values)))
        done = np.zeros((len(b_values), len(a_values)), dtype=bool)

    t0 = time.time()
    n_max_t = max(measure_times)
    cells_done_run = 0
    for ia, a in enumerate(a_values):
        timed_out = False
        for ib, b in enumerate(b_values):
            if done[ib, ia]:
                continue
            if time_budget and (time.time() - t0) > time_budget:
                timed_out = True
                break
            harv_acc, eps_acc, alpha_acc = [], [], []
            for rep in range(n_replicates):
                res = run_simulation(
                    n_steps=n_max_t,
                    a=float(a), b=float(b),
                    rule='maximum',
                    neighborhood='euclidean',
                    include_self=False,
                    f_random=0.05,
                    snapshot_steps=measure_times,
                    seed=seed + rep * 53 + ia * 7 + ib * 11,
                )
                harv_acc.append(res['harvest_trajectory'][-5:].mean())
                for t in measure_times:
                    s = res['snapshots'][t]
                    eps_acc.append(correlation_length(s))
                    ps = patch_sizes(s)
                    if ps.size > 30:
                        alpha, _, _, _ = fit_powerlaw_clauset(ps)
                        if not np.isnan(alpha) and alpha > 0:
                            alpha_acc.append(alpha)
            H_grid[ib, ia] = float(np.mean(harv_acc))
            eps_grid[ib, ia] = float(np.mean(eps_acc))
            if alpha_acc:
                alpha_grid[ib, ia] = float(np.mean(alpha_acc))
            done[ib, ia] = True
            cells_done_run += 1
        if timed_out:
            break

    if cache_path:
        np.savez(cache_path,
                 H=H_grid, alpha=alpha_grid, eps=eps_grid, done=done,
                 a=a_values, b=b_values)

    print(f'  {cells_done_run} cells finished this run, '
          f'total {int(done.sum())}/{done.size}, '
          f'elapsed {time.time()-t0:.1f}s')
    return dict(a=a_values, b=b_values, H=H_grid,
                alpha=alpha_grid, eps=eps_grid, done=done)


def render(result, save_path='fig4_phase_diagram.png'):
    a_values = result['a']
    b_values = result['b']
    H_grid = result['H']
    alpha_grid = result['alpha']
    eps_grid = result['eps']

    # Light Gaussian smoothing of the noisy maps so the visual
    # trend along the b/a~20 diagonal is easier to see.  The unsmoothed
    # values are still printed in the diagonal table below.
    def _smooth(grid, sigma=1.0):
        from scipy.ndimage import gaussian_filter
        finite = np.isfinite(grid)
        g = np.where(finite, grid, 0.0)
        w = finite.astype(float)
        return gaussian_filter(g, sigma=sigma, mode='nearest') / np.clip(
            gaussian_filter(w, sigma=sigma, mode='nearest'), 1e-9, None)

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 14.0))
    extent = [a_values[0], a_values[-1], b_values[0], b_values[-1]]
    a_line = np.linspace(a_values[0], a_values[-1], 50)

    # (A) Harvest
    ax = axes[0]
    im = ax.imshow(_smooth(H_grid), origin='lower', aspect='auto', extent=extent,
                   cmap='RdYlBu_r', vmin=np.nanmin(H_grid), vmax=5.0)
    ax.set_title('A. Average harvest H', fontsize=13)
    ax.set_xlabel('Pest stress a'); ax.set_ylabel('Water stress b')
    ax.plot(a_line, 20 * a_line, 'w--', lw=1.5, label='b/a = 20')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper left', fontsize=10)
    plt.colorbar(im, ax=ax, label='H')

    # (B) Exponent alpha
    ax = axes[1]
    alpha_plot = _smooth(alpha_grid)
    im = ax.imshow(alpha_plot, origin='lower', aspect='auto', extent=extent,
                   cmap='RdYlBu_r', vmin=0.7, vmax=1.6)
    ax.set_title('B. Power-law exponent alpha (CCDF tail)', fontsize=13)
    ax.set_xlabel('Pest stress a'); ax.set_ylabel('Water stress b')
    ax.plot(a_line, 20 * a_line, 'w-', lw=2.5, label='b/a = 20 (paper criticality)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper left', fontsize=10)
    plt.colorbar(im, ax=ax, label='alpha')

    # (C) Correlation length -- use a robust upper bound for the colormap
    # so the ridge along the diagonal is visible even when the absolute
    # values are smaller than the paper's (paper plots ~600 m, our model
    # is in lattice units of L=100).
    ax = axes[2]
    eps_plot = _smooth(eps_grid)
    vmax_eps = np.nanpercentile(eps_grid, 95)
    vmin_eps = np.nanpercentile(eps_grid, 5)
    im = ax.imshow(eps_plot, origin='lower', aspect='auto', extent=extent,
                   cmap='RdYlBu_r', vmin=vmin_eps, vmax=vmax_eps)
    ax.set_title('C. Correlation length epsilon (lattice units)', fontsize=13)
    ax.set_xlabel('Pest stress a'); ax.set_ylabel('Water stress b')
    ax.plot(a_line, 20 * a_line, 'w-', lw=2.5, label='b/a = 20 (paper criticality)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper left', fontsize=10)
    plt.colorbar(im, ax=ax, label='epsilon')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')

    # Print diagonal alpha statistics
    print('\nalpha along the b/a~20 diagonal:')
    for ia, a in enumerate(a_values):
        target_b = 20 * a
        if b_values.min() <= target_b <= b_values.max():
            ib = int(np.argmin(np.abs(b_values - target_b)))
            print(f'  a={a:.3f}, b={b_values[ib]:.2f} '
                  f'(b/a={b_values[ib]/a:.1f}): '
                  f'alpha={alpha_grid[ib, ia]:.2f}, '
                  f'eps={eps_grid[ib, ia]:.1f}, '
                  f'H={H_grid[ib, ia]:.2f}')


def run(n_replicates=4, seed=2024, save_path='fig4_phase_diagram.png',
        cache_path='phase_diagram_state.npz', time_budget=None):
    """Run the sweep (with optional caching) and render the figure.

    If `time_budget` is provided and the sweep cannot finish within it, the
    figure will only be partially populated; rerun the script to resume.
    """
    print(f'Running phase diagram: {len(DEFAULT_A)}x{len(DEFAULT_B)} grid, '
          f'{n_replicates} replicates...')
    result = sweep(n_replicates=n_replicates, seed=seed,
                   cache_path=cache_path, time_budget=time_budget)
    if not result['done'].all():
        print(f'NOT all cells done ({int(result["done"].sum())}/{result["done"].size}); '
              'rerun this script to resume.')
    render(result, save_path=save_path)
    return result


if __name__ == '__main__':
    import sys
    tb = float(sys.argv[1]) if len(sys.argv) > 1 else None
    run(time_budget=tb)
