import numpy as np
import squidpy as sq


def most_enriched_pair(adata):
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", n_jobs=1, show_progress_bar=False, seed=0)
    z = adata.uns["cluster_nhood_enrichment"]["zscore"].copy()
    cats = adata.obs["cluster"].cat.categories.tolist()
    np.fill_diagonal(z, -np.inf)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    return cats[i], cats[j], z[i, j]


# train: visium_hne
adata_train = sq.datasets.visium_hne_adata()
print("train:", most_enriched_pair(adata_train))

# test: visium_fluo
adata_test = sq.datasets.visium_fluo_adata()
print("test:", most_enriched_pair(adata_test))
