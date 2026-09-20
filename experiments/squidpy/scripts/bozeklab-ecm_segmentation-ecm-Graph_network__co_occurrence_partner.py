"""Confirms the answer for task 'co_occurrence_partner' (train + test variants).

Train: imc dataset -> at the shortest distance interval, which type most co-occurs
with T cells (other than T cells themselves).
Test: mibitof dataset, point16 field of view -> same question for Imm_other cells.
"""
import numpy as np
import squidpy as sq


def top_co_occurring_partner(adata, cluster_key, target):
    occ, interval = sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=True)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    i = cats.index(target)
    row0 = occ[i, :, 0].copy()
    row0[i] = -np.inf
    j = np.argmax(row0)
    return cats[j], row0[j]


# --- train: imc, reference = T cells ---
adata = sq.datasets.imc()
print("train (imc, T cells):", top_co_occurring_partner(adata, "cell type", "T cells"))

# --- test: mibitof, point16, reference = Imm_other ---
adata2 = sq.datasets.mibitof()
sub = adata2[adata2.obs["library_id"] == "point16"].copy()
sub.obs["Cluster"] = sub.obs["Cluster"].cat.remove_unused_categories()
print("test (mibitof point16, Imm_other):", top_co_occurring_partner(sub, "Cluster", "Imm_other"))
