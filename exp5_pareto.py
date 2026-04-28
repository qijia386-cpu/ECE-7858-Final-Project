"""
Experiment 5 -- verify the paper's Pareto-optimality claim.

Lansing et al. claim that under the maximum-harvest decision rule the system
approaches "approximate Pareto optimality":

  > Harvests tend to increase and equalize, approaching Pareto optimality
  > at the phase transition where both the frequency distribution of
  > synchronized irrigation patches and the correlations between them
  > become power laws.

We verify this in three complementary ways.

  (1) Within-rule comparison at a=0.5, b=10.0 (b/a=20, the critical line):
      track mean H_i and std H_i across the lattice as a function of t for
      each of the four decision rules.  Pareto optimality predicts that the
      "maximum" rule both *raises the mean* and *shrinks the std*.

  (2) Histogram of H_i at convergence under each rule.  Maximum should be
      narrow and rightmost; random/minority should be broad and left.

  (3) Lorenz curve and Gini coefficient of H_i over the lattice at
      convergence.  Lower Gini = more equal harvests = closer to Pareto.

The expected result: maximum-rule harvests are simultaneously largest *and*
most equal, consistent with approach to Pareto optimality.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation import (run_simulation, harvest, DEFAULTS,
                        update_maximum, update_majority,
                        update_random, update_minority)


def _gini(x):
    """Gini coefficient of a non-negative array.  0 = perfect equality."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.nan
    # shift so all values are positive (Gini for negative values is ill-defined)
    if x.min() < 0:
        x = x - x.min() + 1e-9
    x_sorted = np.sort(x)
    n = x.size
    cum = np.cumsum(x_sorted)
    # Gini = (2 * sum_i (i * x_i) - (n+1) * sum_x) / (n * sum_x)
    return (2.0 * np.sum((np.arange(1, n + 1)) * x_sorted)
            - (n + 1) * cum[-1]) / (n * cum[-1])


def _harvest_field(state, a, b, neighborhood='euclidean'):
    return harvest(state, a, b,
                   H0=DEFAULTS['H0'], m=DEFAULTS['m'], r=DEFAULTS['r'],
                   n_colors=DEFAULTS['N'], neighborhood=neighborhood)


def run(seed=2024, n_steps=120, n_replicates=4, a=0.5, b=10.0,
        save_path='fig5_pareto.png'):
    rules = ['maximum', 'majority', 'random', 'minority']
    colors = {'maximum': 'tab:blue', 'majority': 'tab:red',
              'random': 'tab:pink', 'minority': 'tab:green'}

    # --- Run all rules and record (a) mean H_i, (b) std H_i, (c) final H_i field
    means = {r: [] for r in rules}    # shape (n_replicates, n_steps+1)
    stds = {r: [] for r in rules}
    final_fields = {r: [] for r in rules}
    final_gini = {r: [] for r in rules}

    for rule in rules:
        for rep in range(n_replicates):
            res = run_simulation(
                n_steps=n_steps,
                a=a, b=b,
                rule=rule,
                neighborhood='euclidean',
                include_self=False,
                f_random=0.05,
                snapshot_steps=list(range(0, n_steps + 1)),
                seed=seed + 17 * rep,
            )
            mean_traj, std_traj = [], []
            for t in range(n_steps + 1):
                Hf = _harvest_field(res['snapshots'][t], a, b)
                mean_traj.append(Hf.mean())
                std_traj.append(Hf.std())
            means[rule].append(mean_traj)
            stds[rule].append(std_traj)

            Hf_final = _harvest_field(res['final_state'], a, b)
            final_fields[rule].append(Hf_final.ravel())
            final_gini[rule].append(_gini(Hf_final))

    # --- Plot
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (a) mean H over time
    ax1 = fig.add_subplot(gs[0, 0])
    for rule in rules:
        m = np.array(means[rule]).mean(axis=0)
        ax1.plot(np.arange(n_steps + 1), m, color=colors[rule],
                 lw=2, label=rule)
    ax1.set_xlabel('Simulation step t')
    ax1.set_ylabel(r'$\langle H_i \rangle$  (mean over lattice)')
    ax1.set_title('(a) Mean harvest over time')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # (b) std H over time -- Pareto: maximum should have *smallest* std
    ax2 = fig.add_subplot(gs[0, 1])
    for rule in rules:
        s = np.array(stds[rule]).mean(axis=0)
        ax2.plot(np.arange(n_steps + 1), s, color=colors[rule],
                 lw=2, label=rule)
    ax2.set_xlabel('Simulation step t')
    ax2.set_ylabel(r'$\sigma(H_i)$  (std over lattice)')
    ax2.set_title('(b) Harvest dispersion over time\n'
                  '(lower = more equal, closer to Pareto)')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # (c) histogram of final H_i
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-2, 5, 60)
    for rule in rules:
        all_H = np.concatenate(final_fields[rule])
        ax3.hist(all_H, bins=bins, density=True, color=colors[rule],
                 alpha=0.4, label=f'{rule} (mean={all_H.mean():.2f})')
    ax3.set_xlabel(r'$H_i$  (per-farmer harvest at convergence)')
    ax3.set_ylabel('density')
    ax3.set_title('(c) Distribution of H_i at convergence')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # (d) Lorenz curves of final H_i
    ax4 = fig.add_subplot(gs[1, 0])
    for rule in rules:
        H_all = np.sort(np.concatenate(final_fields[rule]))
        # shift to non-negative for Lorenz
        if H_all.min() < 0:
            H_all = H_all - H_all.min() + 1e-9
        cum = np.cumsum(H_all)
        cum = cum / cum[-1]
        x = np.linspace(0, 1, H_all.size)
        ax4.plot(x, cum, color=colors[rule], lw=2,
                 label=f'{rule}, Gini={np.mean(final_gini[rule]):.3f}')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect equality')
    ax4.set_xlabel('cumulative share of farmers')
    ax4.set_ylabel('cumulative share of harvest')
    ax4.set_title('(d) Lorenz curves\n(closer to diagonal = more equal)')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(alpha=0.3)

    # (e) Pareto front: mean vs std at convergence (each replicate is a
    #    point).  The "maximum" rule should sit at high mean / low std.
    ax5 = fig.add_subplot(gs[1, 1])
    for rule in rules:
        m_conv = np.array(means[rule])[:, -10:].mean(axis=1)
        s_conv = np.array(stds[rule])[:, -10:].mean(axis=1)
        ax5.scatter(s_conv, m_conv, color=colors[rule], s=70,
                    edgecolor='black', label=rule, zorder=3)
    ax5.set_xlabel(r'$\sigma(H_i)$  at convergence')
    ax5.set_ylabel(r'$\langle H_i \rangle$  at convergence')
    ax5.set_title('(e) Equity-efficiency frontier\n'
                  '(upper-left = closer to Pareto optimum)')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    # an arrow toward "Pareto optimum"
    ax5.annotate('Pareto direction', xy=(0.05, 0.95), xycoords='axes fraction',
                 xytext=(0.45, 0.55), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=10)

    # (f) Gini bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    rule_means_gini = [np.mean(final_gini[r]) for r in rules]
    rule_means_H = [np.mean(np.array(means[r])[:, -10:]) for r in rules]
    rule_colors = [colors[r] for r in rules]
    bars = ax6.bar(rules, rule_means_gini, color=rule_colors,
                   edgecolor='black')
    for bar, g, h in zip(bars, rule_means_gini, rule_means_H):
        ax6.text(bar.get_x() + bar.get_width() / 2, g + 0.005,
                 f'Gini={g:.3f}\nmean H={h:.2f}',
                 ha='center', fontsize=9)
    ax6.set_ylabel('Gini coefficient of H_i')
    ax6.set_title('(f) Inequality of harvest at convergence\n(0 = perfect equality)')
    ax6.set_ylim(0, max(rule_means_gini) * 1.4)
    ax6.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'Fig. 5 (new): Pareto-optimality verification at a={a}, b={b} '
        f'(critical line b/a={b/a:.0f})',
        fontsize=13, y=0.995,
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')

    # ----- print summary -----
    print('  Convergence summary (mean over last 10 steps, mean over replicates):')
    print(f'  {"rule":<10s} | {"mean H":>8s} | {"std H":>8s} | {"Gini":>6s}')
    for rule in rules:
        mH = np.mean(np.array(means[rule])[:, -10:])
        sH = np.mean(np.array(stds[rule])[:, -10:])
        gH = np.mean(final_gini[rule])
        print(f'  {rule:<10s} | {mH:>8.3f} | {sH:>8.3f} | {gH:>6.3f}')
    return dict(means=means, stds=stds, gini=final_gini)


if __name__ == '__main__':
    run()
