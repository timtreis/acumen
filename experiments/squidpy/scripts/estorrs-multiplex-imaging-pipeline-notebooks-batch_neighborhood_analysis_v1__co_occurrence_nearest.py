"""Confirmation script for task id: co_occurrence_nearest (train + test).

Analysis: compute spatial co-occurrence probability between cell type
clusters across increasing distance thresholds, then find which other
cluster has the highest co-occurrence score with a given cell type at
the shortest distance interval.
"""
import squidpy as sq
import pandas as pd


def top_co_occurring(adata, cluster_key, target):
    occ, interval = sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=True)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    ti = cats.index(target)
    vals = pd.Series(occ[ti, :, 0], index=cats).drop(index=target)
    print(vals.sort_values(ascending=False))
    return vals.idxmax()


a = sq.datasets.imc()

# --- train: target = "T cells" ---
train_answer = top_co_occurring(a, "cell type", "T cells")
print("TRAIN ANSWER:", train_answer)

# --- test: target = "endothelial" ---
test_answer = top_co_occurring(a, "cell type", "endothelial")
print("TEST ANSWER:", test_answer)
