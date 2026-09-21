import numpy as np
import squidpy as sq


def top_pair(adata):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", n_jobs=1, seed=0, show_progress_bar=False)
    z = np.array(adata.uns["cluster_nhood_enrichment"]["zscore"], copy=True)
    cats = list(adata.obs["cluster"].cat.categories)
    np.fill_diagonal(z, -np.inf)
    idx = np.unravel_index(np.argmax(z), z.shape)
    return cats[idx[0]], cats[idx[1]], z[idx]


if __name__ == "__main__":
    # train: visium_hne data
    adata_hne = sq.datasets.visium_hne_adata()
    print("train (visium_hne):", top_pair(adata_hne))

    # test: visium_fluo data
    adata_fluo = sq.datasets.visium_fluo_adata()
    print("test (visium_fluo):", top_pair(adata_fluo))
