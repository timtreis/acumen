"""Ground-truth script for task mibitof_colocalization.

Goal: which pair of annotated cell types shows the strongest neighborhood
enrichment (most significant spatial co-localization) in a given imaging
region (point) of the mibitof dataset.

train -> point16
test  -> point23
"""

import numpy as np
import squidpy as sq

adata = sq.datasets.mibitof()

for point_id in ["point16", "point23"]:
    sub = adata[adata.obs["library_id"] == point_id].copy()
    sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(sub, cluster_key="Cluster", seed=0, n_jobs=1, show_progress_bar=False)

    zscore = sub.uns["Cluster_nhood_enrichment"]["zscore"]
    cats = sub.obs["Cluster"].cat.categories
    zz = zscore.copy()
    np.fill_diagonal(zz, -np.inf)
    i, j = np.unravel_index(np.argmax(zz), zz.shape)

    print(point_id)
    print("TOP PAIR:", cats[i], "&", cats[j], "z =", zz[i, j])
    print()
