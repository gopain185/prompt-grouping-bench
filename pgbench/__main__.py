"""python -m pgbench [outdir]: run every experiment, print the summaries and save the raw tables."""

import sys
import time
from pathlib import Path

from . import experiments as ex


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, summary in [("exp1_layouts", ex.layouts, ex.summarise_layouts),
                              ("exp2_grouping", ex.grouping, ex.summarise_grouping),
                              ("exp3_prior_shift", ex.prior_shift, ex.summarise_prior),
                              ("exp4_promise", ex.promise, ex.summarise_promise)]:
        t = time.time()
        df = fn()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ({time.time() - t:.0f}s)\n{summary(df).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
