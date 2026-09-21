import numpy as np
import squidpy as sq

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

cats = adata.obs["cell type"].cat.categories.tolist()

for target in ["macrophages", "CK+ HR+ tumor cell"]:
    zscores = []
    for _ in range(6):
        sq.gr.nhood_enrichment(adata, cluster_key="cell type", show_progress_bar=False)
        z = adata.uns["cell type_nhood_enrichment"]["zscore"]
        zscores.append(z.copy())
    z_mean = np.mean(zscores, axis=0)
    i = cats.index(target)
    row = z_mean[i].copy()
    row[i] = -np.inf
    j = int(np.argmax(row))
    print(f"target={target!r} -> most enriched neighbor = {cats[j]!r} (mean z={row[j]:.2f})")
