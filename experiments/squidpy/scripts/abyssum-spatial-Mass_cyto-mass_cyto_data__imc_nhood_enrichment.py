import os
os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import squidpy as sq

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, seed=0, show_progress_bar=False)

cats = list(adata.obs["cell type"].cat.categories)
z = adata.uns["cell type_nhood_enrichment"]["zscore"]

for target in ["T cells", "macrophages"]:
    i = cats.index(target)
    row = z[i].copy()
    row[i] = -np.inf
    j = np.argmax(row)
    print(f"{target} -> strongest neighborhood enrichment with {cats[j]} (z={row[j]:.2f})")
