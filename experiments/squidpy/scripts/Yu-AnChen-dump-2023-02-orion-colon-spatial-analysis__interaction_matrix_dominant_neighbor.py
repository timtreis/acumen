"""Confirms the answer to the interaction_matrix_dominant_neighbor task (train + test)."""
import squidpy as sq
import numpy as np


def dominant_neighbor_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key, normalized=True)
    m = adata.uns[f"{cluster_key}_interactions"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    best, best_val = None, -np.inf
    for i in range(len(cats)):
        for j in range(len(cats)):
            if i != j and m[i, j] > best_val:
                best_val = m[i, j]
                best = (cats[i], cats[j])
    return best, best_val


# --- train: imc dataset ---
adata_train = sq.datasets.imc()
pair_train, val_train = dominant_neighbor_pair(adata_train, "cell type")
print("TRAIN answer (subject, dominant neighbor):", pair_train, val_train)

# --- test: seqfish dataset ---
adata_test = sq.datasets.seqfish()
pair_test, val_test = dominant_neighbor_pair(adata_test, "celltype_mapped_refined")
print("TEST answer (subject, dominant neighbor):", pair_test, val_test)
