# When Does Test-Time Prompt Grouping Help?

In authorship attribution, every passage answers some prompt, and the prompt leaves its own mark on the text. If
several test passages answer the same prompt, a decoder can remove that shared mark (centring), or forbid two of them
from getting the same author (Hungarian assignment). A grouped cross-validation on a train set where every prompt is
answered by every author promises large gains from this. On the public leaderboard of *Name That LLM*, the same tricks
gained much less or lost (one forum report: +0.10 in grouped CV, then 0.353 to 0.344 on the public board).

This notebook simulates the problem with a known truth: 18 authors in 6 families, a shared prompt effect, and test pools
with a controlled number of answers per prompt. It measures what each decoder gains when the groups are true, when they
are recovered by clustering, and when author frequencies are uneven.

The decoders:

- `plain`: each passage on its own.
- `center`: subtract the group mean of the scores, then decode as if the prompt effect were gone (the common heuristic).
- `hungarian`: plain scores, then at most one passage per author inside each group.
- `joint`: decode each member against the prompt effect estimated from the other members (iterated conditional modes
  under the true model).
- `joint_no_repeat`: `joint` plus the one-passage-per-author constraint.
- `plain_sinkhorn`: plain scores, then rescale until every author holds the same share of the probability mass.

Everything runs on CPU in about 15 to 25 minutes. The package source is embedded in the next cell, so the notebook is
self-contained.
