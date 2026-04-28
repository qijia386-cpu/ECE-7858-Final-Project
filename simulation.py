"""
Adaptive self-organization model of Balinese rice terraces.
Faithful re-implementation of Lansing et al. (2017) PNAS 114(25):6504-6509.

Model definition (from the paper):
  - L x L lattice, L=100, N=4 irrigation schedules (colors).
  - Initial state: every site picks one of 4 schedules uniformly at random.
  - Harvest:
        H^i(t+1) = H_0 - a/(0.1 + f_p^i(t)) - b * f_w^i(t)
    where:
        H_0 = 5,
        f_p^i(t) = fraction of neighbours within a radius r=2 (default:
                   Euclidean disk, 12 cells; Moore neighbourhood, 24 cells,
                   is also supported) that share i's colour at time t,
        f_w^i(t) = fraction of ALL lattice sites with i's colour at time t.
  - Decision rule (synchronous update):
        Each farmer compares his harvest with those of his four nearest
        neighbours (von Neumann) and copies the irrigation schedule of the
        best (or majority/random/minority).
  - Random non-conformity: a small fraction f=0.05 of sites are reset to random
        schedules each step (paper Fig. 4 caption).
  - Boundary conditions: periodic (paper does not specify; periodic is standard
        for SOC lattice models and avoids edge artefacts).

The model proceeds through trial-and-error adaptation; after a transient the
patch-size distribution and the correlation function follow power laws when
b/a is near ~20 (the critical region).
"""

import numpy as np
from scipy.ndimage import convolve

# ---------------------------------------------------------------------------
# Default parameters from the paper
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    L=100,           # lattice side
    N=4,             # number of irrigation schedules
    H0=5.0,          # baseline harvest
    r=2,             # neighbourhood radius for pest interaction
    m=0.1,           # constant in pest-stress denominator
    f_random=0.05,   # fraction of farmers acting non-conformist each step
    a=0.5,           # pest-stress weight (Fig. 3 reference value)
    b=9.6,           # water-stress weight (Fig. 3 reference value)
)


def _moore_kernel(r):
    """Moore-neighborhood (Chebyshev) kernel of radius r, center zeroed.
    Sums to (2r+1)^2 - 1."""
    k = np.ones((2 * r + 1, 2 * r + 1), dtype=np.int32)
    k[r, r] = 0
    return k


def _euclidean_kernel(r):
    """Euclidean-disk kernel of radius r, center zeroed. Counts cells with
    sqrt(dx^2 + dy^2) <= r. For r=2 this gives 12 cells."""
    k = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.int32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx <= r * r:
                k[r + dy, r + dx] = 1
    return k


def _make_kernel(r, neighborhood='euclidean'):
    if neighborhood == 'moore':
        return _moore_kernel(r)
    elif neighborhood == 'euclidean':
        return _euclidean_kernel(r)
    else:
        raise ValueError(f'unknown neighborhood: {neighborhood}')


def _pest_fraction(state, n_colors, kernel, n_neighbors):
    """f_p^i for every site i: fraction of radius-r neighbours sharing i's
    colour, where the neighbourhood is defined by `kernel`."""
    # For each colour c, build a 0/1 mask, convolve with kernel to get the
    # number of neighbours of that colour for every site.
    same_color_count = np.zeros_like(state, dtype=np.float64)
    for c in range(n_colors):
        mask = (state == c).astype(np.int32)
        # 'wrap' = periodic boundary conditions
        nb_c = convolve(mask, kernel, mode='wrap')
        # at each site i, the contribution is nb_c[i] when state[i] == c
        same_color_count += np.where(state == c, nb_c, 0)
    return same_color_count / n_neighbors


def _water_fraction(state, n_colors):
    """f_w^i for every site i: fraction of ALL sites sharing i's colour."""
    total = state.size
    f_w_per_color = np.array(
        [np.sum(state == c) / total for c in range(n_colors)]
    )
    return f_w_per_color[state]


def harvest(state, a, b, H0=5.0, m=0.1, r=2, n_colors=4, neighborhood='euclidean'):
    """Compute H^i for every site under the current state."""
    kernel = _make_kernel(r, neighborhood)
    n_neighbors = int(kernel.sum())
    f_p = _pest_fraction(state, n_colors, kernel, n_neighbors)
    f_w = _water_fraction(state, n_colors)
    return H0 - a / (m + f_p) - b * f_w


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------
def _shift(arr, dy, dx):
    """Periodic shift; returns array of same shape with values from (i+dy, j+dx)."""
    return np.roll(arr, shift=(-dy, -dx), axis=(0, 1))


def _nearest_neighbor_stack(values):
    """Return a (5, L, L) stack: [self, N, S, W, E] using periodic BCs.

    'N' here = neighbour to the north (i-1, j), accessed by shifting (-1, 0).
    """
    return np.stack([
        values,
        _shift(values, -1, 0),  # north
        _shift(values, +1, 0),  # south
        _shift(values, 0, -1),  # west
        _shift(values, 0, +1),  # east
    ], axis=0)


def update_maximum(state, H, include_self=False):
    """Adopt the schedule of whichever cell among {[self], 4 neighbours}
    achieved the highest harvest. With include_self=False (default), the
    update follows the paper text literally: 'copies the irrigation schedule
    of one or more neighbours' -- i.e. always copy a neighbour, never keep
    your own colour in this step. (Random non-conformity is applied
    afterwards in the main loop.)"""
    H_stack = _nearest_neighbor_stack(H)        # (5, L, L): self,N,S,W,E
    C_stack = _nearest_neighbor_stack(state)
    if not include_self:
        H_stack = H_stack[1:]
        C_stack = C_stack[1:]
    best = np.argmax(H_stack, axis=0)
    L = state.shape[0]
    ii, jj = np.indices((L, L))
    return C_stack[best, ii, jj]


def update_majority(state, H, rng):
    """Adopt the most common schedule among {self, 4 neighbours} (random tie-break)."""
    C_stack = _nearest_neighbor_stack(state)    # (5, L, L)
    n_colors = int(state.max()) + 1
    counts = np.zeros((n_colors,) + state.shape, dtype=np.int32)
    for c in range(n_colors):
        counts[c] = np.sum(C_stack == c, axis=0)
    # Random tie-breaking: add tiny random jitter so argmax breaks ties uniformly.
    jitter = rng.random(counts.shape)
    return np.argmax(counts + 0.5 * jitter, axis=0)


def update_random(state, H, rng):
    """Copy the schedule of a randomly chosen neighbour (uniform over 4)."""
    C_stack = _nearest_neighbor_stack(state)[1:]  # 4 neighbours, drop self
    L = state.shape[0]
    pick = rng.integers(0, 4, size=state.shape)
    ii, jj = np.indices((L, L))
    return C_stack[pick, ii, jj]


def update_minority(state, H, rng):
    """Adopt the least common schedule among {self, 4 neighbours} (random tie-break)."""
    C_stack = _nearest_neighbor_stack(state)
    n_colors = int(state.max()) + 1
    counts = np.zeros((n_colors,) + state.shape, dtype=np.float64)
    for c in range(n_colors):
        counts[c] = np.sum(C_stack == c, axis=0)
    # We want minimum count, but we should NOT pick a colour that has count 0
    # (= absent from the neighbourhood); the natural read of "minority" is the
    # rarest colour that is actually present.
    counts[counts == 0] = np.inf
    jitter = rng.random(counts.shape)
    return np.argmin(counts + 0.5 * jitter, axis=0)


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------
def run_simulation(
    n_steps,
    a=DEFAULTS['a'],
    b=DEFAULTS['b'],
    L=DEFAULTS['L'],
    n_colors=DEFAULTS['N'],
    H0=DEFAULTS['H0'],
    r=DEFAULTS['r'],
    m=DEFAULTS['m'],
    f_random=DEFAULTS['f_random'],
    rule='maximum',
    neighborhood='euclidean',
    include_self=False,
    snapshot_steps=None,
    seed=None,
    record_harvest=True,
):
    """Run the model for n_steps and return harvest trajectory + snapshots.

    Parameters
    ----------
    rule : {'maximum', 'majority', 'random', 'minority'}
    neighborhood : {'euclidean', 'moore'}
        Shape of the radius-r neighbourhood used in f_p. Paper says
        "within a radius r" which we interpret as Euclidean (12 cells at r=2).
    include_self : bool
        Whether the update rule's argmax includes the site itself. Paper text
        says farmer copies a neighbour, so default is False.
    snapshot_steps : iterable of int or None
    seed : int or None
    """
    rng = np.random.default_rng(seed)
    state = rng.integers(0, n_colors, size=(L, L), dtype=np.int8)

    snapshots = {0: state.copy()}
    if snapshot_steps is None:
        snapshot_steps = []
    snapshot_steps = set(snapshot_steps)

    H = harvest(state, a, b, H0=H0, m=m, r=r, n_colors=n_colors,
                neighborhood=neighborhood)
    traj = [H.mean()] if record_harvest else []

    for t in range(1, n_steps + 1):
        if rule == 'maximum':
            new_state = update_maximum(state, H, include_self=include_self)
        elif rule == 'majority':
            new_state = update_majority(state, H, rng)
        elif rule == 'random':
            new_state = update_random(state, H, rng)
        elif rule == 'minority':
            new_state = update_minority(state, H, rng)
        else:
            raise ValueError(f'Unknown rule: {rule}')

        if f_random > 0:
            mask = rng.random(state.shape) < f_random
            random_colors = rng.integers(0, n_colors, size=state.shape, dtype=np.int8)
            new_state = np.where(mask, random_colors, new_state).astype(np.int8)

        state = new_state
        H = harvest(state, a, b, H0=H0, m=m, r=r, n_colors=n_colors,
                    neighborhood=neighborhood)
        if record_harvest:
            traj.append(H.mean())
        if t in snapshot_steps:
            snapshots[t] = state.copy()

    return dict(
        harvest_trajectory=np.array(traj),
        snapshots=snapshots,
        final_state=state,
        params=dict(a=a, b=b, L=L, rule=rule, f_random=f_random,
                    H0=H0, r=r, m=m, n_steps=n_steps, seed=seed,
                    neighborhood=neighborhood, include_self=include_self),
    )