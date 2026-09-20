import numpy as np
import squidpy as sq

adata = sq.datasets.merfish()

# Restrict to the Bregma -29 tissue section (batch '0'), used for both variants.
sub = adata[adata.obs["batch"] == "0"].copy()
sub.obs["Cell_class"] = sub.obs["Cell_class"].cat.remove_unused_categories()

sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
sq.gr.co_occurrence(sub, cluster_key="Cell_class")

res = sub.uns["Cell_class_co_occurrence"]
occ = res["occ"]  # occ[i, j, k] = p(exp=j | cond=i) / p(exp=j) at distance interval k
cats = list(sub.obs["Cell_class"].cat.categories)

for label, ref in [("train", "Microglia"), ("test", "Excitatory")]:
    i = cats.index(ref)
    row = occ[i].copy()
    row[i, :] = -np.inf  # exclude the reference type itself
    max_per_other = np.nanmax(row, axis=1)
    j = int(np.argmax(max_per_other))
    print(label, "reference:", ref, "-> partner:", cats[j], "ratio:", max_per_other[j])
