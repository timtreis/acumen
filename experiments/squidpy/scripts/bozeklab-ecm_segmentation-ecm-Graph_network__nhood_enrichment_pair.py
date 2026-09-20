"""Confirms the answer for task 'nhood_enrichment_pair' (train + test variants).

Train: imc dataset, cluster_key='cell type' -> most enriched (highest z-score) pair.
Test: mibitof dataset, point8 field of view, cluster_key='Cluster' -> most enriched pair.
"""
import numpy as np
import squidpy as sq


def top_enriched_pair(adata, cluster_key, seed=0):
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, seed=seed, show_progress_bar=False)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    iu = np.triu_indices_from(z, k=1)
    vals = z[iu]
    k = np.argmax(vals)
    i, j = iu[0][k], iu[1][k]
    return cats[i], cats[j], vals[k]


# --- train: imc ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)
print("train (imc):", top_enriched_pair(adata, "cell type"))

# --- test: mibitof, point8 ---
adata2 = sq.datasets.mibitof()
sub = adata2[adata2.obs["library_id"] == "point8"].copy()
sub.obs["Cluster"] = sub.obs["Cluster"].cat.remove_unused_categories()
sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=10)
print("test (mibitof point8):", top_enriched_pair(sub, "Cluster"))
