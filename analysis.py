"""
Analysis tools for the lattice model.

Implements the quantitative measures used in Lansing et al. (2017):
  - Patch identification: 4-connected components of equal-colour cells.
  - Cumulative patch-size distribution P(>s).
  - Power-law exponent alpha (CCDF tail exponent).
  - Mutual-information correlation function C(d), Eq. 1 in the paper.
  - Correlation length epsilon, Eq. 2.
"""

import numpy as np
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def patch_sizes(state, n_colors=4):
    """Return an array of patch sizes (number of cells per 4-connected
    component) over all colors."""
    sizes = []
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.int8)  # 4-connectivity
    for c in range(n_colors):
        mask = (state == c)
        if not mask.any():
            continue
        labeled, n_components = label(mask, structure=structure)
        if n_components == 0:
            continue
        # bincount[0] is background, drop it
        bc = np.bincount(labeled.ravel())[1:]
        sizes.append(bc)
    if not sizes:
        return np.array([], dtype=np.int64)
    return np.concatenate(sizes)


def ccdf(sizes):
    """Empirical complementary CDF.

    Returns
    -------
    s_unique : ndarray
        Distinct sizes in increasing order.
    P_geq : ndarray
        P(S >= s_k) = (number of observations >= s_k) / N.

    Note: for a continuous power-law tail, P(S > s) and P(S >= s) coincide
    in the large-N limit; we use the >= form because it is non-zero at the
    largest observation and therefore well-defined in log-space.
    """
    sizes = np.asarray(sizes)
    s_sorted = np.sort(sizes)
    n = s_sorted.size
    s_unique, idx = np.unique(s_sorted, return_index=True)
    P_geq = 1.0 - idx / n
    return s_unique, P_geq


def fit_powerlaw_mle(sizes, s_min=None):
    """Maximum-likelihood exponent for a (continuous) power-law tail.

    Model:  p(s) propto s^(-(alpha+1))   for s >= s_min   (so that CCDF ~ s^-alpha)
    Hill estimator:
        alpha_hat = N / sum_i ln(s_i / s_min)
    """
    sizes = np.asarray(sizes, dtype=float)
    if s_min is None:
        s_min = max(np.min(sizes), 1.0)
    tail = sizes[sizes >= s_min]
    if tail.size < 5:
        return np.nan, s_min, tail.size
    alpha = tail.size / np.sum(np.log(tail / s_min))
    return alpha, s_min, tail.size


def fit_powerlaw_clauset(sizes, s_min_grid=None, min_tail=30):
    """Clauset/Shalizi/Newman-style MLE: scan candidate s_min values, fit a
    Hill estimator for each, and pick the s_min that minimises the
    Kolmogorov-Smirnov distance between the empirical CCDF and the
    fitted power law CCDF P(>s) = (s/s_min)^(-alpha).

    Returns
    -------
    (alpha, s_min, n_tail, ks_distance)
    """
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes >= 1]
    if sizes.size < min_tail:
        return np.nan, np.nan, sizes.size, np.nan
    if s_min_grid is None:
        # log-spaced grid up to ~max/3 so that we always have a reasonable tail
        s_max_for_grid = max(2.0, sizes.max() / 3.0)
        s_min_grid = np.unique(np.round(
            np.logspace(0, np.log10(s_max_for_grid), 25)
        ).astype(int))
    best = (np.inf, np.nan, np.nan, 0)  # (ks, alpha, s_min, n_tail)
    for sm in s_min_grid:
        tail = sizes[sizes >= sm]
        if tail.size < min_tail:
            continue
        alpha = tail.size / np.sum(np.log(tail / sm))
        if alpha <= 0:
            continue
        s_sorted = np.sort(tail)
        ecdf = np.arange(1, tail.size + 1) / tail.size       # P(<=s)
        cdf_fit = 1.0 - (s_sorted / sm) ** (-alpha)          # P(<=s) fitted
        ks = float(np.max(np.abs(ecdf - cdf_fit)))
        if ks < best[0]:
            best = (ks, float(alpha), float(sm), tail.size)
    ks, alpha, sm, ntail = best
    return alpha, sm, ntail, ks


# ---------------------------------------------------------------------------
# Mutual-information correlation function (Eq. 1 of the paper)
# ---------------------------------------------------------------------------
def _shannon_entropy(state, n_colors=4):
    """N = -sum_X P_0(X) log2 P_0(X)  (the normalisation constant in Eq. 1)."""
    counts = np.array([np.sum(state == c) for c in range(n_colors)], dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _joint_distribution_at_distance(state, d, n_colors=4):
    """Pd(X, Y): joint distribution of colours at site i and a site at lattice
    distance d. We use horizontal *and* vertical shifts of d cells (periodic
    boundary), so each site contributes 2 pairs at distance d. This matches
    the spirit of an isotropic 2D correlation while keeping the computation
    simple and well-defined.
    """
    L = state.shape[0]
    # horizontal pairs: (i, j) and (i, j+d)
    hor = (state, np.roll(state, -d, axis=1))
    # vertical pairs
    ver = (state, np.roll(state, -d, axis=0))
    X = np.concatenate([hor[0].ravel(), ver[0].ravel()])
    Y = np.concatenate([hor[1].ravel(), ver[1].ravel()])

    # build joint histogram
    P = np.zeros((n_colors, n_colors), dtype=float)
    for x in range(n_colors):
        # for each x, count y values
        y_for_x = Y[X == x]
        if y_for_x.size == 0:
            continue
        for y in range(n_colors):
            P[x, y] = np.sum(y_for_x == y)
    P /= P.sum()
    return P


def correlation_function(state, distances, n_colors=4):
    """Compute C(d) (Eq. 1) for each d in `distances`.

    C(d) = (1/N) * sum_{X,Y} P_d(X,Y) * log2[ P_d(X,Y) / (P_d(X) P_d(Y)) ]

    where N = -sum_X P_0(X) log2 P_0(X) is the Shannon entropy of the
    overall colour distribution. With this normalisation C(0) = 1.
    """
    N = _shannon_entropy(state, n_colors)
    if N == 0:
        return np.zeros(len(distances))

    Cs = []
    for d in distances:
        P = _joint_distribution_at_distance(state, d, n_colors)
        Px = P.sum(axis=1)
        Py = P.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mi = 0.0
            for x in range(n_colors):
                for y in range(n_colors):
                    if P[x, y] > 0 and Px[x] > 0 and Py[y] > 0:
                        mi += P[x, y] * np.log2(P[x, y] / (Px[x] * Py[y]))
        Cs.append(mi / N)
    return np.array(Cs)


def correlation_length(state, n_colors=4, max_d=None):
    """epsilon = sqrt( sum_d d^2 C(d) / sum_d C(d) )  (Eq. 2).

    Sums over d = 1, ..., max_d (default: L/2).  C(d) values are clipped at 0
    from below (mutual information should be >= 0 in theory, but finite-sample
    noise can produce small negatives that would break the sqrt).
    """
    L = state.shape[0]
    if max_d is None:
        max_d = L // 2
    distances = np.arange(1, max_d + 1)
    C = correlation_function(state, distances, n_colors)
    C = np.clip(C, 0.0, None)
    num = np.sum(distances ** 2 * C)
    den = np.sum(C)
    if den <= 0:
        return 0.0
    return np.sqrt(num / den)