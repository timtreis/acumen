import numpy as np
import squidpy as sq

# ---- TRAIN: visium_hne_adata, target region "Hippocampus" ----
adata = sq.datasets.visium_hne_adata()
sq.gr.co_occurrence(adata, cluster_key="cluster")
occ = adata.uns["cluster_co_occurrence"]["occ"]
cats = adata.obs["cluster"].cat.categories.tolist()

target = cats.index("Hippocampus")
row = occ[target]  # (n_clusters, n_intervals)
maxvals = row.max(axis=1)
order = np.argsort(-maxvals)
print("TRAIN dataset: visium_hne_adata, target cluster: Hippocampus")
for k in order[:6]:
    print(cats[k], maxvals[k])

# ---- TEST: visium_fluo_adata_crop, same target region "Hippocampus" ----
adata2 = sq.datasets.visium_fluo_adata_crop()
sq.gr.co_occurrence(adata2, cluster_key="cluster")
occ2 = adata2.uns["cluster_co_occurrence"]["occ"]
cats2 = adata2.obs["cluster"].cat.categories.tolist()

target2 = cats2.index("Hippocampus")
row2 = occ2[target2]
maxvals2 = row2.max(axis=1)
order2 = np.argsort(-maxvals2)
print("\nTEST dataset: visium_fluo_adata_crop, target cluster: Hippocampus")
for k in order2[:6]:
    print(cats2[k], maxvals2[k])
