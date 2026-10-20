"""Test-time decoders for the scores of sim.py, and ways to recover the prompt groups.

Every decoder takes the score matrix and a group id per passage (singletons have their own id) and
returns predicted labels. They know the world's parameters (mu, tau, sigma): the point is to measure what
each decision rule can gain under a given test layout, not to estimate the parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

from .sim import K, World


def _loglik(s: np.ndarray, mu: np.ndarray, var: float) -> np.ndarray:
    """log N(s; mu_y, var I) per class, up to a constant: shape (n, K)."""
    return -((s[:, None, :] - mu[None]) ** 2).sum(-1) / (2 * var)


def _members(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    cuts = np.flatnonzero(np.diff(groups[order])) + 1
    return [g for g in np.split(order, cuts) if len(g) > 1]


def plain(s: np.ndarray, groups: np.ndarray, w: World) -> np.ndarray:
    """Each passage on its own: the prompt pull is just extra noise."""
    return _loglik(s, w.mu, w.sigma**2 + w.tau**2).argmax(1)


def center(s: np.ndarray, groups: np.ndarray, w: World) -> np.ndarray:
    """Subtract the group mean of the scores, then decode as if the pull were gone (the usual heuristic)."""
    s = s.copy()
    for g in _members(groups):
        s[g] -= s[g].mean(0)
    mu_c = w.mu - w.mu.mean(0)  # the group mean also removes about this much of the author signal
    return _loglik(s, mu_c, w.sigma**2).argmax(1)


def hungarian(s: np.ndarray, groups: np.ndarray, w: World) -> np.ndarray:
    """Plain log-likelihoods, then at most one passage per author inside each group."""
    ll = _loglik(s, w.mu, w.sigma**2 + w.tau**2)
    pred = ll.argmax(1)
    for g in _members(groups):
        if len(g) <= K:
            rows, cols = linear_sum_assignment(-ll[g])
            pred[g[rows]] = cols
    return pred


def joint(s: np.ndarray, groups: np.ndarray, w: World, no_repeat: bool = False, sweeps: int = 4) -> np.ndarray:
    """Approximate joint MAP under the model: each member is decoded against the prompt pull estimated
    from the other members' residuals (iterated conditional modes); optionally one passage per author."""
    t2, s2 = w.tau**2, w.sigma**2
    pred = plain(s, groups, w)
    for g in _members(groups):
        m = len(g)
        sg, yg = s[g], pred[g].copy()
        for _ in range(sweeps):
            r = sg - w.mu[yg]
            tot = r.sum(0)
            # Posterior of the pull given the other m-1 members, then the predictive of member j.
            shrink = t2 / (s2 + (m - 1) * t2)
            u_hat = shrink * (tot[None] - r)
            var = s2 + t2 * s2 / (s2 + (m - 1) * t2)
            ll = -((sg[:, None, :] - u_hat[:, None, :] - w.mu[None]) ** 2).sum(-1) / (2 * var)
            if no_repeat and m <= K:
                rows, cols = linear_sum_assignment(-ll)
                new = np.empty(m, int)
                new[rows] = cols
            else:
                new = ll.argmax(1)
            if np.array_equal(new, yg):
                break
            yg = new
        pred[g] = yg
    return pred


def sinkhorn(logp: np.ndarray, target: np.ndarray | None = None, iters: int = 100) -> np.ndarray:
    """Rescale class columns until each class holds the target share of the probability mass."""
    p = np.exp(logp - logp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    t = np.full(p.shape[1], 1 / p.shape[1]) if target is None else target
    for _ in range(iters):
        p *= (t * len(p) / p.sum(0))[None]
        p /= p.sum(1, keepdims=True)
    return p


def plain_balanced(s: np.ndarray, groups: np.ndarray, w: World) -> np.ndarray:
    return sinkhorn(_loglik(s, w.mu, w.sigma**2 + w.tau**2)).argmax(1)


DECODERS = {
    "plain": plain,
    "center": center,
    "hungarian": hungarian,
    "joint": joint,
    "joint_no_repeat": lambda s, g, w: joint(s, g, w, no_repeat=True),
    "plain_sinkhorn": plain_balanced,
}


# ----------------------------------------------------------------------------------------------- grouping

def singletons(content: np.ndarray, prompts: np.ndarray) -> np.ndarray:
    return np.arange(len(content))


def oracle(content: np.ndarray, prompts: np.ndarray) -> np.ndarray:
    return prompts.copy()


def cluster(content: np.ndarray, threshold: float, max_size: int = K) -> np.ndarray:
    """Average-linkage clusters on cosine distance, cut at threshold; clusters above max_size become singletons."""
    z = linkage(content, method="average", metric="cosine")
    g = fcluster(z, t=threshold, criterion="distance")
    sizes = np.bincount(g)
    out = g.astype(np.int64)
    big = sizes[g] > max_size
    out[big] = g.max() + 1 + np.flatnonzero(big)
    return out


def pair_quality(groups: np.ndarray, prompts: np.ndarray) -> tuple[float, float]:
    """Precision and recall of same-group pairs as same-prompt pairs."""
    same_g = groups[:, None] == groups[None, :]
    same_p = prompts[:, None] == prompts[None, :]
    off = ~np.eye(len(groups), dtype=bool)
    tp = (same_g & same_p & off).sum()
    pg, pp = (same_g & off).sum(), (same_p & off).sum()
    return (tp / pg if pg else 1.0), (tp / pp if pp else 1.0)
