import squidpy as sq
import numpy as np

# --- train variant: imc dataset ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, show_progress_bar=False)

z = adata.uns["cell type_nhood_enrichment"]["zscore"]
cats = adata.obs["cell type"].cat.categories.tolist()
zz = z.copy()
np.fill_diagonal(zz, -np.inf)
i, j = np.unravel_index(np.argmax(zz), zz.shape)
print("TRAIN (imc) most enriched pair:", cats[i], "and", cats[j], "z=", round(zz[i, j], 2))

# --- test variant: seqfish dataset ---
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2, coord_type="generic", delaunay=True)
sq.gr.nhood_enrichment(
    adata2, cluster_key="celltype_mapped_refined", n_jobs=1, show_progress_bar=False
)

z2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"]
cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
zz2 = z2.copy()
np.fill_diagonal(zz2, -np.inf)
i2, j2 = np.unravel_index(np.argmax(zz2), zz2.shape)
print("TEST (seqfish) most enriched pair:", cats2[i2], "and", cats2[j2], "z=", round(zz2[i2, j2], 2))
