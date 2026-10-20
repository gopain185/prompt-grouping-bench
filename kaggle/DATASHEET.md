# Prompt Grouping Simulations

Simulated authorship-attribution scores where **every passage's true author and true prompt are known**, together
with the result tables of a benchmark of test-time decoders that use prompt groups: centring, Hungarian assignment,
joint decoding with a shared prompt effect, and Sinkhorn class balancing.

## Provenance

Everything is simulated by the `pgbench` package (repository `prompt-grouping-bench` by Loc Nguyen, MIT), seed
`20261021`. The companion notebook embeds the full source and regenerates every table. No real texts, people or
model outputs are involved; the 18 "authors" and their family structure only mirror the layout of the Kaggle
competition *Name That LLM* (6 families of sizes 5, 4, 5, 2, 1, 1).

## How the data is generated

Passage `i`, by author `y` in answer to prompt `p`, gets an 18-dimensional score vector

`s_i = mu[y] + u[p] + e_i`, with `u[p] ~ N(0, 0.7^2 I)` shared by all answers to prompt `p` and `e_i ~ N(0, I)`.

`mu[y]` is `1.0 * onehot(y)` plus `2.0` spread evenly over the classes of `y`'s family, so the family is easy and the
author inside it is hard (a plain decoder reaches about 0.35 balanced accuracy). Each passage also has a 32-dimensional
unit content vector `c_i = v[p] + 0.6 * noise / sqrt(32)`, normalised, which is the only thing a grouping method sees.
A prompt with `m` answers has `m` distinct authors drawn at random.

## Files

| file | rows | content |
| --- | --- | --- |
| `test_pools.csv` | about 24,000 | one simulated test pool of about 4,000 passages per layout (`m` = 1, 2, 3, 6, 18 and `mixed`) |
| `exp1_layouts.csv` | 840 | balanced accuracy per layout, replicate and decoder, with the true prompt groups |
| `exp2_grouping.csv` | 2,700 | the same with groups recovered by clustering, for three content-noise levels and three cut-offs |
| `exp3_prior_shift.csv` | 480 | plain decoding against Sinkhorn balancing when author frequencies are uneven, for an unbiased classifier and one that carries an unbalanced training prior |
| `exp4_promise.csv` | 600 | the gain measured on a train-like layout (150 prompts x 18 authors) against test layouts |

`test_pools.csv` columns: `layout`, `row_id`, `prompt`, `author` (0-17; 0-4, 5-8, 9-13, 14-15, 16 and 17 are the six
families), `score_0` ... `score_17`, `content_0` ... `content_31`.

## License

CC0 1.0 (public domain dedication).

## Intended use

Testing and teaching decision rules that use test-time groups of related samples: how much they can gain, how that
gain depends on how many samples a group really has, and what recovering the groups wrongly costs. Also a check for
cross-validation designs whose layout differs from the test set's.

## Known limitations

- The score model is Gaussian with a shared, isotropic prompt effect. Real classifiers output log-probabilities whose
  prompt effect is neither Gaussian nor isotropic, so the sizes of the gains are illustrative; the directions and the
  dependence on the layout are the point.
- The decoders are given the true model parameters (`mu`, `tau`, `sigma`). Estimating them adds error not modelled here.
- Grouping uses one content representation and average linkage; real pipelines mix lexical and embedding similarities.
