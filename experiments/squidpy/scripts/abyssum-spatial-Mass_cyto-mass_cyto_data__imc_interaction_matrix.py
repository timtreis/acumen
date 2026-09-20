import os
os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import squidpy as sq

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.interaction_matrix(adata, cluster_key="cell type")

cats = list(adata.obs["cell type"].cat.categories)
im = adata.uns["cell type_interactions"]

for target in ["endothelial", "macrophages"]:
    i = cats.index(target)
    row = im[i].astype(float).copy()
    row[i] = -1
    j = np.argmax(row)
    print(f"{target} -> most spatial interactions with {cats[j]} (count={row[j]:.0f})")
