"""
Experiment 3 -- replicate the model curves of Fig. 1C and 1D.

At the critical-line parameters a=0.5, b=10.0 (b/a = 20.0) we measure the
patches-state observables averaged over many replicates and over a small
window of time steps where the system sits in the power-law-patch attractor.

  (C) Cumulative patch-size distribution P(>s).  Paper reports CCDF tail
      exponent alpha around 0.93 (Gianyar) and ~ 1.0 in their model.
  (D) Correlation function C(d) using the mutual-information definition
      (Eq. 1).  Paper reports approximate power-law decay over the
      "scaling region" of moderate d.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation import run_simulation
from analysis import (patch_sizes, ccdf, fit_powerlaw_clauset,
                      correlation_function, correlation_length)


def _log_binned_pdf(sizes, n_bins=20):
    """Return (bin centres, density) for a log-binned PDF.  Useful for
    visualising the tail of a power law without singleton-noise dominating
    the small-s region."""
    sizes = np.asarray(sizes)
    sizes = sizes[sizes >= 1]
    bins = np.unique(np.round(
        np.logspace(0, np.log10(sizes.max() + 1), n_bins + 1)
    ).astype(int))
    if bins.size < 2:
        return np.array([]), np.array([])
    hist, edges = np.histogram(sizes, bins=bins)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    density = hist / (widths * sizes.size)
    return centres, density


def run(n_replicates=12,
        measure_times=(25, 30, 35, 40, 45, 50),
        a=0.5, b=10.0,
        seed=2024,
        save_path='fig1CD_model.png'):
    distances = np.unique(np.round(np.logspace(0, np.log10(50), 24)).astype(int))

    all_sizes = []          # for CCDF (pooled over replicates and times)
    all_alpha = []          # per-snapshot Clauset alpha
    all_eps = []            # per-snapshot correlation length
    C_per_snapshot = []     # per-snapshot correlation function

    for rep in range(n_replicates):
        res = run_simulation(
            n_steps=max(measure_times),
            a=a, b=b,
            rule='maximum',
            neighborhood='euclidean',
            include_self=False,
            f_random=0.05,
            snapshot_steps=measure_times,
            seed=seed + 31 * rep,
        )
        for t in measure_times:
            state = res['snapshots'][t]
            ps = patch_sizes(state)
            all_sizes.append(ps)
            alpha, sm, ntail, ks = fit_powerlaw_clauset(ps)
            all_alpha.append(alpha)
            all_eps.append(correlation_length(state))
            C_per_snapshot.append(correlation_function(state, distances))

    sizes_pooled = np.concatenate(all_sizes)
    print(f'  pooled #patches = {len(sizes_pooled)} '
          f'(min={sizes_pooled.min()}, max={sizes_pooled.max()})')

    s_unique, P_geq = ccdf(sizes_pooled)

    # per-snapshot statistics
    alpha_arr = np.array([x for x in all_alpha if not np.isnan(x)])
    eps_arr = np.array(all_eps)
    print(f'  Clauset alpha across snapshots: '
          f'mean={alpha_arr.mean():.3f}, std={alpha_arr.std():.3f}, '
          f'median={np.median(alpha_arr):.3f}')
    print(f'  correlation length eps (lattice units): '
          f'mean={eps_arr.mean():.2f}, std={eps_arr.std():.2f}')
    print(f'  Paper reports alpha ~ 0.93 (Gianyar) and around 1.0 in their model')

    # Pooled CCDF MLE -- constrain s_min grid to the physically meaningful
    # power-law regime (we don't want it picking the finite-size tail
    # roll-off above s ~ 200, which has a different scaling).
    s_min_grid = np.unique(np.round(np.logspace(0, np.log10(80), 25)).astype(int))
    alpha_pool, sm_pool, n_tail_pool, ks_pool = fit_powerlaw_clauset(
        sizes_pooled, s_min_grid=s_min_grid)
    print(f'  Pooled Clauset MLE (s_min in [1,80]): alpha={alpha_pool:.3f}, '
          f's_min={sm_pool:.0f}, n_tail={n_tail_pool}, KS={ks_pool:.3f}')

    # Average correlation function (clip negatives from MI noise)
    C_mean = np.mean(np.stack(C_per_snapshot), axis=0)

    # ----- plot -----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (C) CCDF
    ax = axes[0]
    # Show the CCDF as the headline curve, with reference power-law slope -1
    ax.loglog(s_unique, P_geq, 's', markersize=4.5, color='tab:blue',
              alpha=0.55, mfc='none', mec='tab:blue', mew=1.0,
              label=f'model (a={a}, b={b}), pooled CCDF')
    # Reference slope anchored in the visible scaling region
    s_ref = np.logspace(np.log10(max(sm_pool, 5)), np.log10(400), 50)
    P_anchor = (1 - np.searchsorted(np.sort(sizes_pooled),
                                    s_ref[0]) / sizes_pooled.size)
    ax.loglog(s_ref, P_anchor * (s_ref / s_ref[0]) ** (-alpha_pool),
              'k--', lw=2.0,
              label=f'fitted: alpha = {alpha_pool:.2f} (s_min={sm_pool:.0f}, '
                    f'KS={ks_pool:.2f})')
    # also show paper's reference exponent for visual comparison
    ax.loglog(s_ref, P_anchor * (s_ref / s_ref[0]) ** (-1.0),
              'r:', lw=1.6, label='paper reference: alpha = 1.0')
    ax.set_xlim(0.9, max(2000, sizes_pooled.max() * 1.2))
    ax.set_ylim(max(1.0/sizes_pooled.size/2, 1e-5), 1.5)
    ax.set_xlabel('Patch size s', fontsize=12)
    ax.set_ylabel('P(>s)', fontsize=12)
    ax.set_title(
        f'Fig. 1C replication: cumulative patch-size distribution\n'
        f'per-snapshot Clauset alpha = '
        f'{alpha_arr.mean():.2f} +/- {alpha_arr.std():.2f}',
        fontsize=11)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3, which='both')

    # (D) C(d)
    ax = axes[1]
    pos = C_mean > 0
    ax.loglog(distances[pos], C_mean[pos], 'bs-', markersize=5,
              label=f'model (a={a}, b={b}), mean over snapshots\n'
                    f'eps = {eps_arr.mean():.1f} +/- {eps_arr.std():.1f} '
                    f'lattice units')
    ax.axvline(eps_arr.mean(), color='gray', linestyle=':', lw=2,
               label=f'epsilon = {eps_arr.mean():.1f}')
    # Fit a slope on the early scaling region (d <= ~ epsilon)
    eps_mean = eps_arr.mean()
    idx_fit = np.where(
        (distances >= 1) & (distances <= max(2, 0.8 * eps_mean)) & pos
    )[0]
    if len(idx_fit) >= 2:
        slope, intercept = np.polyfit(np.log10(distances[idx_fit]),
                                      np.log10(C_mean[idx_fit]), 1)
        d_fit = np.logspace(np.log10(distances[idx_fit].min()),
                            np.log10(distances[idx_fit].max()), 30)
        ax.loglog(d_fit, 10 ** intercept * d_fit ** slope, 'r--', lw=1.8,
                  label=f'fitted slope on d <= 0.8 eps: {slope:.2f}')
    ax.set_xlabel('Distance d (lattice units)', fontsize=12)
    ax.set_ylabel('C(d)  (mutual information)', fontsize=12)
    ax.set_title('Fig. 1D replication: correlation function (mutual info)',
                 fontsize=11)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')
    return dict(sizes=sizes_pooled, alpha_pool=alpha_pool,
                alpha_per=alpha_arr, eps=eps_arr,
                distances=distances, C=C_mean)


if __name__ == '__main__':
    run()
