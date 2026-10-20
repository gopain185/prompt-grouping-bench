"""The four experiments. Each returns one row per (setting, replicate, decoder)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import DECODERS, cluster, oracle, pair_quality, singletons, sinkhorn, _loglik
from .sim import K, World, balanced_accuracy, layout, sample, skewed_prior, test_pool

SEED = 20261021
N_TEST = 4000
LAYOUTS = [1, 2, 3, 6, 12, 18, "mixed"]


def _score(pool, groups, w, decoders=DECODERS):
    return {name: balanced_accuracy(pool.labels, f(pool.scores, groups, w)) for name, f in decoders.items()}


def layouts(reps: int = 20, world: World = World()) -> pd.DataFrame:
    """Exp 1: oracle prompt groups, test layouts from 1 to 18 authors per prompt."""
    rng = np.random.default_rng(SEED)
    rows = []
    for m in LAYOUTS:
        for r in range(reps):
            pool = test_pool(world, N_TEST, m, rng)
            for name, acc in _score(pool, oracle(pool.content, pool.prompts), world).items():
                rows.append(dict(m=str(m), rep=r, decoder=name, bal_acc=acc))
    return pd.DataFrame(rows)


def grouping(reps: int = 10, ms=(2, 18, "mixed"), etas=(0.3, 0.6, 0.9), thresholds=(0.3, 0.5, 0.7)) -> pd.DataFrame:
    """Exp 2: recovered groups (content clustering) of varying purity, against oracle groups and none."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for eta in etas:
        w = World(eta=eta)
        for m in ms:
            for r in range(reps):
                pool = test_pool(w, N_TEST, m, rng)
                groupings = {"none": singletons(pool.content, pool.prompts), "oracle": oracle(pool.content, pool.prompts)}
                for t in thresholds:
                    groupings[f"cluster@{t}"] = cluster(pool.content, t)
                for gname, g in groupings.items():
                    prec, rec = pair_quality(g, pool.prompts)
                    for name, acc in _score(pool, g, w).items():
                        rows.append(dict(eta=eta, m=str(m), rep=r, grouping=gname, precision=prec, recall=rec,
                                         decoder=name, bal_acc=acc))
    return pd.DataFrame(rows)


def prior_shift(reps: int = 20, strengths=(0.0, 0.5, 1.0, 2.0), m: int = 2, world: World = World()) -> pd.DataFrame:
    """Exp 3: unequal author frequencies in the test pool; balancing assumes they are equal. The classifier is
    either unbiased (its scores are the likelihood) or carries the prior of an unbalanced training pool."""
    rng = np.random.default_rng(SEED + 2)
    train_prior = skewed_prior(1.0, np.random.default_rng(SEED + 20))  # fixed, like a pool of 1,100-3,300 per author
    rows = []
    for st in strengths:
        for r in range(reps):
            prior = skewed_prior(st, rng)
            pool = sample(world, *layout(N_TEST // m, m, rng, prior), rng)
            ll = _loglik(pool.scores, world.mu, world.sigma**2 + world.tau**2)
            tv = 0.5 * np.abs(prior - 1 / K).sum()
            for bias, z in [("none", ll), ("train prior", ll + np.log(train_prior))]:
                preds = {"plain": z.argmax(1), "sinkhorn_uniform": sinkhorn(z).argmax(1),
                         "sinkhorn_true_prior": sinkhorn(z, prior).argmax(1)}
                for name, p in preds.items():
                    rows.append(dict(strength=st, rep=r, prior_tv=tv, classifier_bias=bias, decoder=name,
                                     bal_acc=balanced_accuracy(pool.labels, p)))
    return pd.DataFrame(rows)


def promise(reps: int = 20, world: World = World(eta=0.6), threshold: float = 0.5) -> pd.DataFrame:
    """Exp 4: the gain a train-layout CV promises (150 prompts x 18 authors, as in the competition's train)
    against the gain realised on test pools with fewer authors per prompt, with the same clustering."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for r in range(reps):
        settings = {"cv (18 per prompt, 150 prompts)": sample(world, *layout(150, 18, rng), rng)}
        for m in [1, 2, 3, "mixed"]:
            settings[f"test ({m} per prompt)"] = test_pool(world, N_TEST, m, rng)
        for sname, pool in settings.items():
            g = cluster(pool.content, threshold)
            for name, acc in _score(pool, g, world).items():
                rows.append(dict(setting=sname, rep=r, decoder=name, bal_acc=acc))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- summaries

def _gain(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    base = df[df.decoder == "plain"].set_index(keys + ["rep"])["bal_acc"]
    d = df.join(base.rename("plain_acc"), on=keys + ["rep"])
    d["gain"] = d.bal_acc - d.plain_acc
    out = d.groupby(keys + ["decoder"]).agg(bal_acc=("bal_acc", "mean"), gain=("gain", "mean"),
                                           gain_sd=("gain", "std")).reset_index()
    return out


def summarise_layouts(df):
    t = _gain(df, ["m"]).pivot(index="m", columns="decoder", values="gain")
    t.insert(0, "plain_acc", df[df.decoder == "plain"].groupby("m").bal_acc.mean())
    return t.reindex([str(m) for m in LAYOUTS]).round(4).reset_index()


def summarise_grouping(df):
    q = df.groupby(["eta", "m", "grouping"])[["precision", "recall"]].mean()
    t = _gain(df, ["eta", "m", "grouping"]).pivot(index=["eta", "m", "grouping"], columns="decoder", values="gain")
    return q.join(t).round(4).reset_index()


def summarise_prior(df):
    t = df.groupby(["classifier_bias", "strength", "decoder"]).bal_acc.mean().unstack()
    t = t.join(df.groupby(["classifier_bias", "strength"]).prior_tv.mean())
    return t[["prior_tv", "plain", "sinkhorn_uniform", "sinkhorn_true_prior"]].round(4).reset_index()


def summarise_promise(df):
    return _gain(df, ["setting"]).pivot(index="setting", columns="decoder", values="gain").round(4).reset_index()
