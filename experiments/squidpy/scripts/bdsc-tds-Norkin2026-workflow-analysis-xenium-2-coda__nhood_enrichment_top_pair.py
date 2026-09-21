import numpy as np
import squidpy as sq

# --- train: imc dataset ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1, show_progress_bar=False)
z = adata.uns["cell type_nhood_enrichment"]["zscore"].copy()
cats = adata.obs["cell type"].cat.categories.tolist()
np.fill_diagonal(z, -np.inf)
idx = np.unravel_index(np.argmax(z), z.shape)
print("TRAIN (imc) top enriched pair:", cats[idx[0]], "&", cats[idx[1]], "zscore=", z[idx])

# --- test: seqfish dataset ---
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2)
sq.gr.nhood_enrichment(adata2, cluster_key="celltype_mapped_refined", seed=0, n_jobs=1, show_progress_bar=False)
z2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"].copy()
cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
np.fill_diagonal(z2, -np.inf)
idx2 = np.unravel_index(np.argmax(z2), z2.shape)
print("TEST (seqfish) top enriched pair:", cats2[idx2[0]], "&", cats2[idx2[1]], "zscore=", z2[idx2])
