import os
os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import squidpy as sq

adata = sq.datasets.imc()
cats = list(adata.obs["cell type"].cat.categories)
occ, interval = sq.gr.co_occurrence(adata, cluster_key="cell type", copy=True)


def best_partner(target):
    i = cats.index(target)
    best_j, best_v = None, -np.inf
    for j, b in enumerate(cats):
        if b == target:
            continue
        v = occ[i, j, :].max()
        if v > best_v:
            best_v, best_j = v, j
    return cats[best_j], best_v


for target in ["T cells", "basal CK tumor cell"]:
    partner, v = best_partner(target)
    print(f"{target} -> strongest co-occurrence enrichment with {partner} (ratio={v:.2f})")
