# prompt-grouping-bench

When does it pay to group test samples that share a hidden cause? In authorship attribution every passage answers a
prompt, and the prompt leaves its own mark. Grouping test passages by prompt allows centring, one-author-per-prompt
assignment, or joint decoding. A cross-validation on a train set where every prompt is answered by every author
promises large gains; a test set with few answers per prompt does not deliver them.

This repository simulates the problem with a known truth (18 authors in 6 families, a Gaussian shared prompt
effect, test pools with a controlled number of answers per prompt) and measures six decoders under true groups,
recovered groups and uneven author frequencies.

## Results (20 pools of 4,000 passages per cell unless noted; plain decoding about 0.353)

Gain over plain decoding with the true prompt groups:

| answers per prompt | centre | Hungarian | joint | joint, one per author |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 |
| 2 | -0.067 | +0.006 | +0.023 | +0.024 |
| 3 | +0.002 | +0.013 | +0.038 | +0.042 |
| 6 | +0.066 | +0.029 | +0.063 | +0.071 |
| 18 | +0.101 | +0.117 | +0.086 | +0.115 |

What a train-layout CV promises, against test pools, with the same clustering (cut 0.5):

| setting | centre | Hungarian | joint, one per author |
| --- | --- | --- | --- |
| CV: 150 prompts x 18 authors | +0.098 | +0.118 | +0.111 |
| test, 1 answer per prompt | -0.102 | -0.004 | -0.019 |
| test, 2 answers per prompt | -0.057 | +0.004 | +0.015 |

- Wrong merges (low pair precision) turn every grouping gain negative; centring loses whenever groups are small.
- For balanced accuracy, Sinkhorn balancing to equal class mass removes a training-prior bias (+0.020 to +0.022) and
  costs about 0.003 at most when the test is uneven; balancing to the true uneven frequencies costs up to 0.040.

Full tables and discussion: `kaggle/notebook_findings.md`, or the Kaggle notebook
[When Does Test-Time Prompt Grouping Help?](https://www.kaggle.com/code/locdunhat/when-does-test-time-prompt-grouping-help)
and the dataset [Prompt Grouping Simulations](https://www.kaggle.com/datasets/locdunhat/prompt-grouping-simulations).

## Layout

- `pgbench/sim.py`: the world (author means with family structure, prompt effect, content vectors) and test layouts.
- `pgbench/methods.py`: decoders (`plain`, `center`, `hungarian`, `joint`, `joint_no_repeat`, `plain_sinkhorn`),
  grouping (`oracle`, `cluster`, `singletons`) and pair precision / recall.
- `pgbench/experiments.py`: the four experiments and their summaries.
- `kaggle/`: dataset and notebook builder, datasheet, notebook text.

## Run

```
pip install -r requirements.txt
python -m pytest -q
python -m pgbench results      # about 15 minutes on one CPU core
python kaggle/build_kaggle.py --reuse-results
```

## License

MIT for the code; the generated data is CC0.
