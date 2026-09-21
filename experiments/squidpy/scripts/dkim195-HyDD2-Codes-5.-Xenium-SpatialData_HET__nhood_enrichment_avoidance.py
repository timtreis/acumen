"""
Reproduces the answers for task `nhood_enrichment_avoidance` (train + test).

Mirrors the mined analysis (squidpy.gr.spatial_neighbors + squidpy.gr.nhood_enrichment,
visualized with squidpy.pl.nhood_enrichment / squidpy.pl.spatial_scatter), applied to
the bundled `mibitof` dataset instead of the original (unavailable) Xenium data.
The original script processed several spatial regions of one experiment with the
identical pipeline; here the "region" (mibitof's `library_id`) plays that same role
for the train vs. test variant.
"""

import numpy as np
import squidpy as sq


def most_segregated_pair(adata, cluster_key="Cluster", seed=0):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=seed, show_progress_bar=False)
    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    mat = zscore.copy()
    np.fill_diagonal(mat, np.inf)
    i, j = np.unravel_index(np.argmin(mat), mat.shape)
    pair = sorted([cats[i], cats[j]])
    return f"{pair[0]}, {pair[1]}"


def main():
    adata_full = sq.datasets.mibitof()

    train = adata_full[adata_full.obs["library_id"] == "point23"].copy()
    train_answer = most_segregated_pair(train)
    print("train (point23):", train_answer)

    test = adata_full[adata_full.obs["library_id"] == "point8"].copy()
    test_answer = most_segregated_pair(test)
    print("test (point8):", test_answer)


if __name__ == "__main__":
    main()
