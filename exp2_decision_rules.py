"""
Experiment 2 -- replicate Fig. 3B.

Mean harvest H(t) over the full simulation for the four decision rules:
    maximum, majority, random, minority.
Reference parameters: a=0.5, b=9.6 (paper Fig. 3 caption).

Paper readings (approximate, from Fig. 3B):
    maximum  -> H ~ 2.0  (rapid increase to maximum)
    majority -> H ~ 1.7  (similar rapid increase)
    random   -> H ~ 1.4  (inferior)
    minority -> H ~ 1.0  (no improvement above the initial random level)
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation import run_simulation


def run(seed=2024, n_steps=150, n_replicates=5, save_path='fig3B_decision_rules.png'):
    rules = ['maximum', 'majority', 'random', 'minority']
    colors = {'maximum': 'tab:blue', 'majority': 'tab:red',
              'random': 'tab:pink', 'minority': 'tab:green'}

    trajectories = {}
    for rule in rules:
        traj_runs = []
        for rep in range(n_replicates):
            res = run_simulation(
                n_steps=n_steps,
                a=0.5, b=9.6,
                rule=rule,
                neighborhood='euclidean',
                include_self=False,
                f_random=0.05,
                seed=seed + 17 * rep,
            )
            traj_runs.append(res['harvest_trajectory'])
        trajectories[rule] = np.array(traj_runs)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for rule in rules:
        traj = trajectories[rule]
        mean = traj.mean(axis=0)
        std = traj.std(axis=0)
        t_axis = np.arange(traj.shape[1])
        ax.plot(t_axis, mean, label=rule, color=colors[rule], linewidth=2)
        ax.fill_between(t_axis, mean - std, mean + std,
                        color=colors[rule], alpha=0.15)
    ax.set_xlabel('Simulation step t', fontsize=12)
    ax.set_ylabel('Harvest H', fontsize=12)
    ax.set_title('Fig. 3B replication: decision rules at a=0.5, b=9.6', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0.7, 2.3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')

    # report converged values
    print('  Converged H (last 20 steps, mean over replicates):')
    summary = {}
    for rule in rules:
        h_conv = trajectories[rule][:, -20:].mean()
        h_std = trajectories[rule][:, -20:].std()
        summary[rule] = (h_conv, h_std)
        print(f'    {rule:10s} -> H = {h_conv:.3f} +/- {h_std:.3f}')
    return trajectories, summary


if __name__ == '__main__':
    run()
