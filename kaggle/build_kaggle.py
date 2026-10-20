"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only] [--reuse-results]

writes kaggle/dataset/ (simulated test pools with their true authors, prompts and content vectors, the result tables
of the four experiments and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and
reruns the experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgbench import experiments as ex  # noqa: E402
from pgbench.sim import K, World, test_pool  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/prompt-grouping-simulations"
NOTEBOOK_ID = "locdunhat/when-does-test-time-prompt-grouping-help"
TITLE = "When Does Test-Time Prompt Grouping Help?"


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([ex.SEED, 9])
    parts = []
    w = World()
    for m in [1, 2, 3, 6, 18, "mixed"]:
        pool = test_pool(w, ex.N_TEST, m, rng)
        df = pd.DataFrame(pool.scores.round(5), columns=[f"score_{k}" for k in range(K)])
        df.insert(0, "layout", str(m))
        df.insert(1, "row_id", np.arange(len(df)))
        df.insert(2, "prompt", pool.prompts)
        df.insert(3, "author", pool.labels)
        cont = pd.DataFrame(pool.content.round(5), columns=[f"content_{j}" for j in range(pool.content.shape[1])])
        parts.append(pd.concat([df, cont], axis=1))
    pd.concat(parts).to_csv(out / "test_pools.csv", index=False)
    if rerun:
        for name, fn in [("exp1_layouts", ex.layouts), ("exp2_grouping", ex.grouping),
                         ("exp3_prior_shift", ex.prior_shift), ("exp4_promise", ex.promise)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m pgbench results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Prompt Grouping Simulations",
        "id": DATASET_ID,
        "subtitle": "Authorship scores with known prompts, for testing test-time grouping",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["nlp", "benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "pgbench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/prompt-grouping-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("prompt-grouping-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pgbench import experiments as ex
pd.set_option("display.precision", 4); pd.set_option("display.width", 160)
"""

EXP1 = """t = time.time(); e1 = ex.layouts(); print(round(time.time() - t), "s")
s1 = ex.summarise_layouts(e1); s1
"""

PLOT1 = """order = [m for m in s1.m if m != "mixed"]
x = [int(m) for m in order]
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in ["center", "hungarian", "joint", "joint_no_repeat", "plain_sinkhorn"]:
    ax.plot(x, s1.set_index("m").loc[order, name], marker="o", label=name)
ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_xlabel("authors per prompt in the test pool (oracle groups)"); ax.set_ylabel("gain in balanced accuracy over plain")
ax.legend(); plt.tight_layout(); plt.show()
"""

EXP2 = """t = time.time(); e2 = ex.grouping(); print(round(time.time() - t), "s")
s2 = ex.summarise_grouping(e2); s2
"""

EXP3 = """t = time.time(); e3 = ex.prior_shift(); print(round(time.time() - t), "s")
ex.summarise_prior(e3)
"""

EXP4 = """t = time.time(); e4 = ex.promise(); print(round(time.time() - t), "s")
ex.summarise_promise(e4)
"""

SAMPLE = """import glob
from pgbench.methods import DECODERS, oracle, cluster
from pgbench.sim import World, Pool, balanced_accuracy
hits = glob.glob("/kaggle/input/**/test_pools.csv", recursive=True)
if hits:
    df = pd.read_csv(hits[0]); w = World(); rows = []
    for m, g in df.groupby("layout", sort=False):
        pool = Pool(g.filter(like="score_").to_numpy(), g.author.to_numpy(), g.prompt.to_numpy(), g.filter(like="content_").to_numpy())
        for gname, groups in [("oracle", oracle(pool.content, pool.prompts)), ("cluster@0.5", cluster(pool.content, 0.5))]:
            rows.append({"layout": m, "grouping": gname, **{k: balanced_accuracy(pool.labels, f(pool.scores, groups, w)) for k, f in DECODERS.items()}})
    display(pd.DataFrame(rows))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. How many answers per prompt does grouping need?\n\nThe test pool has `m` authors per "
                     "prompt and the decoders get the true prompt groups. Each cell is the mean gain in balanced "
                     "accuracy over `plain` (20 pools of 4,000 passages per layout). `mixed` draws m from "
                     "{1, 2, 3, 6, 18} per prompt."),
        ("code", EXP1),
        ("code", PLOT1),
        ("markdown", "## 2. Recovered groups instead of true ones\n\nThe groups now come from average-linkage "
                     "clustering of the content vectors, cut at 0.3, 0.5 or 0.7 (clusters above 18 passages are "
                     "split back into singletons). `eta` is the content noise: higher means prompts are harder to "
                     "recover. `precision` and `recall` are for same-group pairs as same-prompt pairs. 10 pools per "
                     "cell."),
        ("code", EXP2),
        ("markdown", "## 3. Balancing when authors are not equally frequent\n\nSinkhorn balancing pushes every "
                     "author to the same share of the probability mass. Here the test pool's author frequencies are "
                     "uneven (`prior_tv` is their total-variation distance from uniform; 2 authors per prompt). The classifier is either unbiased or carries the prior of an unbalanced training pool (`train prior`), as a model trained on a public pool of 1,100 to 3,300 passages per author would."),
        ("code", EXP3),
        ("markdown", "## 4. What a train-layout CV promises\n\nThe competition's train has every prompt answered by "
                     "all 18 authors. A grouped CV on that layout measures grouping gains at m = 18. Here is that "
                     "promise next to the gain on test pools with fewer answers per prompt, with the same "
                     "clustering (cut 0.5)."),
        ("code", EXP4),
        ("markdown", "## 5. The pools in the dataset\n\nThe companion dataset holds one simulated test pool per "
                     "layout, with every passage's true author, prompt and content vector."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "prompt-grouping.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "prompt-grouping.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["nlp", "benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
