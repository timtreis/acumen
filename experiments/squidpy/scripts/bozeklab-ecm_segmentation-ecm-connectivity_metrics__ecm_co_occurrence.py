import numpy as np
import squidpy as sq


def top_cooccurring_partner(adata, target):
    sq.gr.co_occurrence(adata, cluster_key="cell type", show_progress_bar=False)
    res = adata.uns["cell type_co_occurrence"]
    occ = res["occ"]
    cats = adata.obs["cell type"].cat.categories.tolist()
    ti = cats.index(target)
    vals = occ[ti, :, 0].copy()
    vals[ti] = -np.inf
    best = int(np.argmax(vals))
    return cats[best], vals[best]


# train
adata = sq.datasets.imc()
partner, val = top_cooccurring_partner(adata, "macrophages")
print("train (macrophages):", partner, val)

# test
adata = sq.datasets.imc()
partner, val = top_cooccurring_partner(adata, "T cells")
print("test (T cells):", partner, val)
