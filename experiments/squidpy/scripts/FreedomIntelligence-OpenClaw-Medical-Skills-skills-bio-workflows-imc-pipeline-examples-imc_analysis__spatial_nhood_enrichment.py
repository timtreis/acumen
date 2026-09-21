"""
Confirmation script for task id: spatial_nhood_enrichment

Reproduces the answers for both the train and test variants by re-running
squidpy's spatial-neighbor-graph + neighborhood-enrichment permutation test
on two bundled datasets, then reporting the pair of distinct cell
types/clusters with the highest (most attractive) enrichment z-score.
"""

import os

os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import pandas as pd
import squidpy as sq


def top_enriched_pair(adata, cluster_key, library_key=None):
    sq.gr.spatial_neighbors(
        adata, coord_type="generic", delaunay=True, library_key=library_key
    )
    sq.gr.nhood_enrichment(
        adata, cluster_key=cluster_key, n_jobs=1, show_progress_bar=False
    )
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    mask = ~np.eye(len(cats), dtype=bool)
    zz = z.copy()
    zz[~mask] = -np.inf
    i, j = np.unravel_index(np.argmax(zz), zz.shape)
    return cats[i], cats[j], z[i, j]


# --- train variant: imc dataset ---
adata_imc = sq.datasets.imc()
a, b, score = top_enriched_pair(adata_imc, "cell type")
print("TRAIN (imc):", a, "and", b, "z =", round(float(score), 2))

# --- test variant: mibitof dataset (multiple images -> library_key) ---
adata_mibi = sq.datasets.mibitof()
a2, b2, score2 = top_enriched_pair(adata_mibi, "Cluster", library_key="library_id")
print("TEST (mibitof):", a2, "and", b2, "z =", round(float(score2), 2))
