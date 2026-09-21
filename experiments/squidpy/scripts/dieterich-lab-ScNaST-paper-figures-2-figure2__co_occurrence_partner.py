import numpy as np
import squidpy as sq

# --- train: imc dataset, target cluster = "endothelial" ---
adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")
res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]  # shape (n_clusters, n_clusters, n_intervals)
cats = adata.obs["cell type"].cat.categories.tolist()

target = "endothelial"
ti = cats.index(target)
r0 = occ[:, ti, 0].copy()  # co-occurrence at the smallest distance interval
r0[ti] = -np.inf  # exclude self
best = cats[int(np.argmax(r0))]
print("train answer:", best)

# --- test: seqfish dataset, target cluster = "Cardiomyocytes" ---
adata2 = sq.datasets.seqfish()
sq.gr.co_occurrence(adata2, cluster_key="celltype_mapped_refined")
res2 = adata2.uns["celltype_mapped_refined_co_occurrence"]
occ2 = res2["occ"]
cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()

target2 = "Cardiomyocytes"
ti2 = cats2.index(target2)
r0_2 = occ2[:, ti2, 0].copy()
r0_2[ti2] = -np.inf
best2 = cats2[int(np.argmax(r0_2))]
print("test answer:", best2)
