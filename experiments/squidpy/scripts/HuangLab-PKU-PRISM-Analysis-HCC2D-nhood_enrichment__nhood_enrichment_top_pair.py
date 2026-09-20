import numpy as np
import pandas as pd
import squidpy as sq

adata_full = sq.datasets.mibitof()


def top_pair(library_id):
    adata = adata_full[adata_full.obs.library_id == library_id].copy()
    adata.obs["Cluster"] = adata.obs["Cluster"].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key="Cluster", n_perms=1000, seed=0, show_progress_bar=False)
    cats = adata.obs["Cluster"].cat.categories.tolist()
    arr = np.array(adata.uns["Cluster_nhood_enrichment"]["zscore"], dtype=float, copy=True)
    np.fill_diagonal(arr, np.nan)
    z = pd.DataFrame(arr, index=cats, columns=cats)
    stacked = z.stack()
    pair = stacked.idxmax()
    return pair, stacked[pair]


for lib in ["point8", "point16"]:
    pair, score = top_pair(lib)
    print(lib, "->", pair, round(float(score), 3))
