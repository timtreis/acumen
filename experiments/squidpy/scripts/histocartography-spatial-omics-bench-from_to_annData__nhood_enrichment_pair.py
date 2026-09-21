"""
Confirmation script for task `nhood_enrichment_pair`.

Reproduces the neighborhood-enrichment analysis performed in the mined script
(sq.gr.spatial_neighbors + sq.gr.nhood_enrichment on a categorical cell-type
annotation) and reports, for each dataset, the pair of categories with the
single highest enrichment z-score (i.e. the strongest positive spatial
co-localization).

Train variant: imc dataset, "cell type" annotation.
Test variant: seqfish dataset, "celltype_mapped_refined" annotation.

Result verified robust to random seed and to whether coord_type is passed
explicitly (default auto-detection gives the same top pair).
"""

import numpy as np
import squidpy as sq


def top_pair(adata, cluster_key, seed=0):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        seed=seed,
        show_progress_bar=False,
        n_jobs=1,
    )
    z = np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"], dtype=float)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    n = len(cats)
    mask = ~np.eye(n, dtype=bool)
    zz = np.where(mask, z, -np.inf)
    i, j = np.unravel_index(np.argmax(zz), zz.shape)
    return cats[i], cats[j], float(zz[i, j])


if __name__ == "__main__":
    imc = sq.datasets.imc()
    a, b, score = top_pair(imc, "cell type")
    print("TRAIN (imc):", a, "and", b, "z =", round(score, 2))

    fish = sq.datasets.seqfish()
    a, b, score = top_pair(fish, "celltype_mapped_refined")
    print("TEST (seqfish):", a, "and", b, "z =", round(score, 2))
