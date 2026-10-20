## Findings

Numbers are from the tables above (plain decoding reaches about 0.353 balanced accuracy in every setting).

**1. The gain depends on how many answers a prompt really has.** With the true groups, at 18 answers per prompt
Hungarian assignment adds +0.117 and centring +0.101. At 2 answers per prompt Hungarian adds only +0.006 and centring
*loses* 0.067: subtracting a two-passage mean removes half of each passage's own author signal along with the prompt
effect. At 3 answers centring is back to zero. Joint decoding, which estimates the prompt effect from the *other*
members, is never worse than plain with true groups (+0.024 at 2 answers, +0.115 with the constraint at 18).

**2. Wrong merges turn the gains negative.** With recovered groups, what matters is pair precision. At 2 answers per
prompt and a loose cut (precision 0.14), Hungarian loses 0.017 and joint decoding with the constraint 0.022; centring
loses at every cut. The best cut differs by layout: at 18 answers per prompt a tight cut (0.3) misses pairs (recall
0.82, +0.096 for Hungarian instead of +0.122 at cut 0.5), while at 2 answers the cut of 0.5 that is best for the
train-like layout merges too much (precision 0.59). With noisier content (eta 0.9) and 2 answers, the 0.5 cut has
precision 0.33 and every grouping decoder is at or below plain (centring -0.081).

**3. A train-layout CV over-promises.** The competition's train answers every prompt 18 times, so a grouped CV
measures the gain at 18 answers: +0.118 for Hungarian, +0.111 for joint with the constraint, +0.098 for centring. On
test pools with one answer per prompt, the same clustering yields -0.004, -0.019 and -0.102; with two answers +0.004,
+0.015 and -0.057. This is the pattern reported on the forum (a CV gain of about +0.10 that became a small loss on
the public board), and it follows from the layout alone, without any modelling error.

**4. Balance to uniform, never to the true prior, for balanced accuracy.** Balanced accuracy weights every author's
recall equally, so the best decision ignores the author frequencies. For a classifier whose scores are the
likelihood, Sinkhorn balancing to equal mass costs about 0.003 at most, even when the test frequencies are far from equal
(total variation 0.23); balancing to the *true* uneven frequencies costs up to 0.040. For a classifier that carries an
unbalanced training prior, balancing to equal mass recovers +0.020 to +0.022 and comes within about 0.003 of the unbiased
classifier, whatever the test frequencies. So balancing removes a training-prior bias; it is not a bet that the test
is balanced.

**What this means for a real pipeline.** Measure grouping tricks with a CV that subsamples each validation prompt to
the number of answers the test is likely to have (1-3, if the forum reports are right), with the groups recovered the
same way as on the test, and prefer decoders that estimate the shared effect from the other members over plain
centring.

**Limitations.** Gaussian scores with an isotropic prompt effect; decoders that know the true parameters; one content
representation. The sizes are illustrative, the directions are the point.
