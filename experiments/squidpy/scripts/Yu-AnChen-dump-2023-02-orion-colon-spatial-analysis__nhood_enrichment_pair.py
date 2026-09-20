"""Confirms the answer to the nhood_enrichment_pair task (train + test)."""
import squidpy as sq
import numpy as np


def top_enriched_pair(adata, cluster_key, seed=0):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=seed, n_jobs=1, show_progress_bar=False)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    best, best_val = None, -np.inf
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            if z[i, j] > best_val:
                best_val = z[i, j]
                best = (cats[i], cats[j])
    return best, best_val


# --- train: imc dataset ---
adata_train = sq.datasets.imc()
pair_train, val_train = top_enriched_pair(adata_train, "cell type")
print("TRAIN answer:", pair_train, val_train)

# --- test: seqfish dataset ---
adata_test = sq.datasets.seqfish()
pair_test, val_test = top_enriched_pair(adata_test, "celltype_mapped_refined")
print("TEST answer:", pair_test, val_test)
