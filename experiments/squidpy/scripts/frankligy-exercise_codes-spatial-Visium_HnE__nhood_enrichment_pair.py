import numpy as np
import pandas as pd
import squidpy as sq


def top_pair(adata, cluster_key="cluster"):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, n_jobs=1)
    z = np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"], dtype=float)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    zz = z.copy()
    for i in range(zz.shape[0]):
        zz[i, i] = np.nan
    df = pd.DataFrame(zz, index=cats, columns=cats)
    stacked = df.stack()
    return stacked.idxmax(), stacked.max()


if __name__ == "__main__":
    # TRAIN: visium_hne_adata
    adata = sq.datasets.visium_hne_adata()
    pair, val = top_pair(adata)
    print("TRAIN visium_hne_adata max enrichment pair:", pair, val)

    # TEST: visium_fluo_adata
    adata2 = sq.datasets.visium_fluo_adata()
    pair2, val2 = top_pair(adata2)
    print("TEST visium_fluo_adata max enrichment pair:", pair2, val2)
