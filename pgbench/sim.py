"""A known-truth model of authorship attribution when every passage answers a prompt.

Each passage i, written by author y_i (one of K = 18) in answer to prompt p_i, gets a score vector

    s_i = mu[y_i] + u[p_i] + e_i,      u[p] ~ N(0, tau^2 I),  e_i ~ N(0, sigma^2 I)

mu[y] = beta * onehot(y) + beta_family * onehot-of-family(y) spread over the family's classes, so the family
(GPT, Gemini, Claude, GLM, Qwen, Kimi) is easy and the author within a family is hard. u[p] is the prompt's
pull: every answer to the same prompt shares it, which is what test-time grouping tries to remove. Each
passage also gets a content vector c_i = v[p_i] + eta * noise, which is all a grouping method can see.

Train has the competition's layout (every prompt answered by all 18 authors once). The test layout is a
parameter: m authors per prompt, drawn without replacement, optionally with unequal author frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILIES = [5, 4, 5, 2, 1, 1]  # GPT, Gemini, Claude, GLM, Qwen, Kimi
K = sum(FAMILIES)
FAMILY = np.repeat(np.arange(len(FAMILIES)), FAMILIES)


@dataclass(frozen=True)
class World:
    beta: float = 1.0          # author signal
    beta_family: float = 2.0   # family signal, shared by the authors of a family
    tau: float = 0.7           # prompt pull (sd per score dimension)
    sigma: float = 1.0         # passage noise
    content_dim: int = 32
    eta: float = 0.6           # content noise: higher means prompts are harder to recover

    @property
    def mu(self) -> np.ndarray:
        fam = (FAMILY[:, None] == FAMILY[None, :]).astype(float)
        return self.beta * np.eye(K) + self.beta_family * fam / fam.sum(1, keepdims=True)


@dataclass
class Pool:
    scores: np.ndarray    # (n, K)
    labels: np.ndarray    # (n,)
    prompts: np.ndarray   # (n,)
    content: np.ndarray   # (n, content_dim), unit rows


def layout(n_prompts: int, m: int | np.ndarray, rng: np.random.Generator, prior: np.ndarray | None = None):
    """Labels and prompt ids: prompt p gets m (or m[p]) distinct authors, drawn with probability prior."""
    ms = np.broadcast_to(np.asarray(m), (n_prompts,))
    p = np.full(K, 1 / K) if prior is None else prior
    labels, prompts = [], []
    for j, mj in enumerate(ms):
        labels.append(rng.choice(K, size=int(mj), replace=False, p=p))
        prompts.append(np.full(int(mj), j))
    return np.concatenate(labels), np.concatenate(prompts)


def sample(world: World, labels: np.ndarray, prompts: np.ndarray, rng: np.random.Generator) -> Pool:
    n_prompts = prompts.max() + 1
    u = rng.normal(0, world.tau, (n_prompts, K))
    s = world.mu[labels] + u[prompts] + rng.normal(0, world.sigma, (len(labels), K))
    v = rng.normal(size=(n_prompts, world.content_dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    c = v[prompts] + world.eta * rng.normal(size=(len(labels), world.content_dim)) / np.sqrt(world.content_dim)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return Pool(s, labels, prompts, c)


def test_pool(world: World, n: int, m: int | str, rng: np.random.Generator, prior: np.ndarray | None = None) -> Pool:
    """About n test passages with m authors per prompt; m="mixed" draws m from {1, 2, 3, 6, 18}."""
    if m == "mixed":
        ms = rng.choice([1, 2, 3, 6, 18], size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10])
        ms = ms[: np.searchsorted(np.cumsum(ms), n) + 1]
        return sample(world, *layout(len(ms), ms, rng, prior), rng)
    return sample(world, *layout(max(1, n // int(m)), int(m), rng, prior), rng)


def skewed_prior(strength: float, rng: np.random.Generator) -> np.ndarray:
    """Unequal author frequencies: a Dirichlet draw that is more uneven for larger strength; 0 gives uniform."""
    if strength <= 0:
        return np.full(K, 1 / K)
    p = rng.dirichlet(np.full(K, 1 / strength))
    return 0.5 * p + 0.5 / K  # every author keeps at least half the uniform share


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))
