import numpy as np
import squidpy as sq


def top_enriched_pair(adata, cluster_key="cluster"):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, show_progress_bar=False)
    z = np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"])
    cats = adata.obs[cluster_key].cat.categories
    mat = z.copy()
    np.fill_diagonal(mat, -np.inf)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    pair = sorted([cats[idx[0]], cats[idx[1]]])
    return pair, mat[idx]


# train: visium_hne data
adata_train = sq.datasets.visium_hne_adata()
pair_train, score_train = top_enriched_pair(adata_train, cluster_key="cluster")
print("train:", pair_train, score_train)

# test: visium_fluo data
adata_test = sq.datasets.visium_fluo_adata()
pair_test, score_test = top_enriched_pair(adata_test, cluster_key="cluster")
print("test:", pair_test, score_test)
