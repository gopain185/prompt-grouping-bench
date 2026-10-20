import numpy as np
import pytest

from pgbench import experiments as ex
from pgbench.methods import DECODERS, cluster, hungarian, joint, oracle, pair_quality, plain, singletons, sinkhorn
from pgbench.sim import FAMILY, K, World, balanced_accuracy, layout, sample, skewed_prior
from pgbench.sim import test_pool as make_test_pool


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_layout_has_distinct_authors_per_prompt(rng):
    labels, prompts = layout(50, 6, rng)
    assert len(labels) == 300
    for p in range(50):
        assert len(set(labels[prompts == p])) == 6


def test_mu_makes_family_easier_than_author():
    mu = World().mu
    assert mu.shape == (K, K) and len(FAMILY) == K
    d = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    same, other = FAMILY[:, None] == FAMILY[None, :], FAMILY[:, None] != FAMILY[None, :]
    off = ~np.eye(K, dtype=bool)
    assert d[same & off].max() < d[other].min()


def test_mixed_layout_size(rng):
    pool = make_test_pool(World(), 1000, "mixed", rng)
    assert 1000 <= len(pool.labels) <= 1018


def test_singletons_change_nothing(rng):
    w = World()
    pool = make_test_pool(w, 400, 3, rng)
    g = singletons(pool.content, pool.prompts)
    base = plain(pool.scores, g, w)
    for name in ["center", "hungarian", "joint", "joint_no_repeat"]:
        assert np.array_equal(DECODERS[name](pool.scores, g, w), base)


def test_one_author_per_prompt_is_singletons(rng):
    w = World()
    pool = make_test_pool(w, 300, 1, rng)
    g = oracle(pool.content, pool.prompts)
    assert np.array_equal(joint(pool.scores, g, w), plain(pool.scores, g, w))


def test_no_prompt_pull_means_joint_equals_plain(rng):
    w = World(tau=0.0)
    pool = make_test_pool(w, 360, 18, rng)
    g = oracle(pool.content, pool.prompts)
    assert np.array_equal(joint(pool.scores, g, w), plain(pool.scores, g, w))


def test_hungarian_never_repeats_inside_a_group(rng):
    w = World()
    pool = make_test_pool(w, 360, 6, rng)
    g = oracle(pool.content, pool.prompts)
    pred = hungarian(pool.scores, g, w)
    for p in np.unique(g):
        assert len(set(pred[g == p])) == (g == p).sum()


def test_full_groups_help(rng):
    w = World()
    pool = make_test_pool(w, 1800, 18, rng)
    g = oracle(pool.content, pool.prompts)
    base = balanced_accuracy(pool.labels, plain(pool.scores, g, w))
    assert balanced_accuracy(pool.labels, DECODERS["joint_no_repeat"](pool.scores, g, w)) > base + 0.05


def test_sinkhorn_hits_the_target_mass(rng):
    logp = rng.normal(size=(500, K))
    t = skewed_prior(1.0, rng)
    p = sinkhorn(logp, t, iters=300)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-9)
    np.testing.assert_allclose(p.sum(0) / len(p), t, atol=1e-3)


def test_skewed_prior(rng):
    np.testing.assert_allclose(skewed_prior(0, rng), np.full(K, 1 / K))
    p = skewed_prior(2.0, rng)
    assert p.min() >= 0.5 / K - 1e-12 and abs(p.sum() - 1) < 1e-12


def test_cluster_caps_size_and_oracle_is_perfect(rng):
    w = World(eta=0.3)
    pool = make_test_pool(w, 720, 18, rng)
    g = cluster(pool.content, 0.9)
    assert np.bincount(g).max() <= K
    assert pair_quality(oracle(pool.content, pool.prompts), pool.prompts) == (1.0, 1.0)
    prec, rec = pair_quality(cluster(pool.content, 0.5), pool.prompts)
    assert prec > 0.9 and rec > 0.9


def test_balanced_accuracy():
    y = np.array([0, 0, 1, 1, 1, 1])
    assert balanced_accuracy(y, np.array([0, 1, 1, 1, 1, 1])) == pytest.approx(0.75)


def test_experiments_run_small():
    assert len(ex.summarise_layouts(ex.layouts(reps=1))) == len(ex.LAYOUTS)
    assert not ex.summarise_prior(ex.prior_shift(reps=1, strengths=(0.0, 1.0))).empty
    assert not ex.summarise_promise(ex.promise(reps=1)).empty
    s = ex.summarise_grouping(ex.grouping(reps=1, ms=(2,), etas=(0.6,), thresholds=(0.5,)))
    assert set(s.grouping) == {"none", "oracle", "cluster@0.5"}
