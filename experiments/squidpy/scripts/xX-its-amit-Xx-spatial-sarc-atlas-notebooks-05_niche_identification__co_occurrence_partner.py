import numpy as np
import squidpy as sq

# --- train variant: imc dataset, target cell type = endothelial ---
adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")

cats = adata.obs["cell type"].cat.categories.tolist()
occ = adata.uns["cell type_co_occurrence"]["occ"]
target = "endothelial"
tidx = cats.index(target)
row0 = occ[tidx, :, 0].copy()
row0[tidx] = -np.inf
best = cats[int(np.argmax(row0))]
print("TRAIN answer:", best)

# --- test variant: seqfish dataset, target cell type = Cardiomyocytes ---
adata2 = sq.datasets.seqfish()
sq.gr.co_occurrence(adata2, cluster_key="celltype_mapped_refined")

cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
occ2 = adata2.uns["celltype_mapped_refined_co_occurrence"]["occ"]
target2 = "Cardiomyocytes"
tidx2 = cats2.index(target2)
row0b = occ2[tidx2, :, 0].copy()
row0b[tidx2] = -np.inf
best2 = cats2[int(np.argmax(row0b))]
print("TEST answer:", best2)
