import squidpy as sq
import numpy as np

a = sq.datasets.mibitof()

for lib in ["point16", "point23"]:
    sub = a[a.obs.library_id == lib].copy()
    sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=6)
    sq.gr.nhood_enrichment(sub, cluster_key="Cluster", show_progress_bar=False)
    z = sub.uns["Cluster_nhood_enrichment"]["zscore"].copy()
    cats = sub.obs["Cluster"].cat.categories.tolist()
    np.fill_diagonal(z, -np.inf)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    print(lib, "->", cats[i], "and", cats[j], "z =", round(float(z[i, j]), 2))

# train (point16) answer: Tcell_CD8 and Tcell_CD4
# test (point23) answer: Endothelial and Imm_other
