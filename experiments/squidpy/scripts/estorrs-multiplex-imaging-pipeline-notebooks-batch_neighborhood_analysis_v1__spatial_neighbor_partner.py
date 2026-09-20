"""Confirmation script for task id: spatial_neighbor_partner (train + test).

Analysis: build a spatial neighbor graph over cells and compute the
cluster-by-cluster interaction (neighbor count) matrix, then find which
other cluster most frequently borders a given cell type.
"""
import squidpy as sq
import pandas as pd


def top_partner(adata, cluster_key, target):
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
    mat = adata.uns[f"{cluster_key}_interactions"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    df = pd.DataFrame(mat, index=cats, columns=cats)
    row = df.loc[target].drop(index=target)
    print(row.sort_values(ascending=False))
    return row.idxmax()


# --- train: imc dataset, target = "T cells" ---
a_train = sq.datasets.imc()
train_answer = top_partner(a_train, "cell type", "T cells")
print("TRAIN ANSWER:", train_answer)

# --- test: mibitof dataset, target = "Tcell_CD8" ---
a_test = sq.datasets.mibitof()
test_answer = top_partner(a_test, "Cluster", "Tcell_CD8")
print("TEST ANSWER:", test_answer)
