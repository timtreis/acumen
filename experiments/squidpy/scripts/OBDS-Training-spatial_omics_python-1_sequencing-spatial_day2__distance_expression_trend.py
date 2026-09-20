import squidpy as sq
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr

adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)

for cluster, gene, key in [
    ("Hippocampus", "Neurod6", "dist_hip"),
    ("Striatum", "Penk", "dist_str"),
]:
    a = adata.copy()
    sq.tl.var_by_distance(a, groups=cluster, cluster_key="cluster", design_matrix_key=key)
    dist = a.obsm[key][f"{cluster}_raw"].values

    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    expr = a[:, gene].X
    expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()

    r, _ = pearsonr(dist, expr)
    direction = "increases" if r > 0 else "decreases"
    print(cluster, gene, "pearson r =", round(r, 4), "->", direction, "with distance")
