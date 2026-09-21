import numpy as np
import pandas as pd
import squidpy as sq


def top_enriched_pair(adata, cluster_key, library_key=None):
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, library_key=library_key)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, show_progress_bar=False)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"].copy()
    cats = adata.obs[cluster_key].cat.categories.tolist()
    np.fill_diagonal(z, -np.inf)
    df = pd.DataFrame(z, index=cats, columns=cats)
    idx = np.unravel_index(np.argmax(df.values), df.values.shape)
    return cats[idx[0]], cats[idx[1]], df.values[idx]


# TRAIN: imc dataset
adata_train = sq.datasets.imc()
a, b, score = top_enriched_pair(adata_train, cluster_key="cell type")
print("TRAIN answer:", a, "&", b, "z=", score)

# TEST: mibitof dataset (multiple imaging points -> use library_key)
adata_test = sq.datasets.mibitof()
a, b, score = top_enriched_pair(adata_test, cluster_key="Cluster", library_key="library_id")
print("TEST answer:", a, "&", b, "z=", score)
