"""Confirms the answer for task 'interaction_matrix_pair' (train + test variants).

Train: imc dataset, cluster_key='cell type' -> pair with most spatial-neighbor links.
Test: mibitof dataset, point16 field of view, cluster_key='Cluster' -> same question.
"""
import numpy as np
import squidpy as sq


def top_interacting_pair(adata, cluster_key):
    mat = sq.gr.interaction_matrix(adata, cluster_key=cluster_key, copy=True)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    iu = np.triu_indices_from(mat, k=1)
    vals = mat[iu]
    k = np.argmax(vals)
    i, j = iu[0][k], iu[1][k]
    return cats[i], cats[j], vals[k]


# --- train: imc ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)
print("train (imc):", top_interacting_pair(adata, "cell type"))

# --- test: mibitof, point16 ---
adata2 = sq.datasets.mibitof()
sub = adata2[adata2.obs["library_id"] == "point16"].copy()
sub.obs["Cluster"] = sub.obs["Cluster"].cat.remove_unused_categories()
sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=10)
print("test (mibitof point16):", top_interacting_pair(sub, "Cluster"))
