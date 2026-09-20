import numpy as np
import squidpy as sq

# --- train: imc dataset, target cluster = "endothelial" ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, show_progress_bar=False)
res = adata.uns["cell type_nhood_enrichment"]
z = res["zscore"]
cats = adata.obs["cell type"].cat.categories.tolist()

target = "endothelial"
ti = cats.index(target)
row = z[ti].copy()
row[ti] = -np.inf  # exclude self
best = cats[int(np.argmax(row))]
print("train answer:", best)

# --- test: seqfish dataset, target cluster = "Endothelium" ---
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2)
sq.gr.nhood_enrichment(adata2, cluster_key="celltype_mapped_refined", seed=0, show_progress_bar=False)
res2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]
z2 = res2["zscore"]
cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()

target2 = "Endothelium"
ti2 = cats2.index(target2)
row2 = z2[ti2].copy()
row2[ti2] = -np.inf
best2 = cats2[int(np.argmax(row2))]
print("test answer:", best2)
