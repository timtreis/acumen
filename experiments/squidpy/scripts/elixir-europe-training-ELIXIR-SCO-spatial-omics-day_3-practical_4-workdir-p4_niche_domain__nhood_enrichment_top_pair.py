import squidpy as sq
import numpy as np


def top_enriched_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1)

    z = np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"], dtype=float).copy()
    cats = list(adata.obs[cluster_key].cat.categories)

    mat = z.copy()
    np.fill_diagonal(mat, -np.inf)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return cats[idx[0]], cats[idx[1]], mat[idx]


if __name__ == "__main__":
    # train: imc data
    adata_train = sq.datasets.imc()
    print("TRAIN answer:", top_enriched_pair(adata_train, "cell type"))

    # test: mibitof data, sample point23
    adata_test = sq.datasets.mibitof()
    sub_test = adata_test[adata_test.obs["library_id"] == "point23"].copy()
    print("TEST answer:", top_enriched_pair(sub_test, "Cluster"))
