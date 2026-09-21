import numpy as np
import squidpy as sq

# --- train variant: imc dataset, target cell type = macrophages ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1, show_progress_bar=False)

cats = adata.obs["cell type"].cat.categories.tolist()
z = adata.uns["cell type_nhood_enrichment"]["zscore"]
target = "macrophages"
tidx = cats.index(target)
row = z[tidx].copy()
row[tidx] = -np.inf
best = cats[int(np.argmax(row))]
print("TRAIN answer:", best)

# --- test variant: seqfish dataset, target cell type = Endothelium ---
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2, coord_type="generic", n_neighs=6)
sq.gr.nhood_enrichment(adata2, cluster_key="celltype_mapped_refined", seed=0, n_jobs=1, show_progress_bar=False)

cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
z2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"]
target2 = "Endothelium"
tidx2 = cats2.index(target2)
row2 = z2[tidx2].copy()
row2[tidx2] = -np.inf
best2 = cats2[int(np.argmax(row2))]
print("TEST answer:", best2)
